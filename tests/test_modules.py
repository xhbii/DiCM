import pytest
import torch
from torch import nn
from dicm.module.sparse_delta import (
    SparseDeltaParam, compose, disabled, export, install_library, payload_count,
    project, set_mode, sparse_delta_supports,
)
from dicm.module.library import ModuleLibrary


def bank(indices, values, shape=(2, 3)):
    return {'weight': dict(shape=shape, indices=torch.tensor(indices, dtype=torch.long),
                           values=torch.tensor(values, dtype=torch.float32))}


def test_global_budget_ties_locks_and_zero():
    mods = {n: SparseDeltaParam(torch.zeros(2, 3)) for n in ('a','b')}
    for mod in mods.values():
        mod.delta.data.fill_(1)
    locks = {n: torch.zeros(2, 3, dtype=torch.bool) for n in mods}
    locks['a'][0,0] = True
    result = project(mods, 3, locks)
    assert result['nonzero'] == 3
    assert mods['a'].delta[0,0] == 0
    assert payload_count(export(mods)) == 3
    assert project(mods, 0, locks) == {'nonzero': 0, 'norm': 0.0}


@pytest.mark.parametrize('budget', [-1, 1.5, True])
def test_reject_invalid_budget(budget):
    with pytest.raises(ValueError):
        project({}, budget, None)


def test_sparse_composition_overlap_cancellation_and_empty():
    a = bank([0,1], [1,2])
    b = bank([1,4], [-2,3])
    ab = compose([a,b])
    assert ab['weight']['indices'].tolist() == [0,4]
    assert ab['weight']['values'].tolist() == [1,3]
    assert torch.equal(ab['weight']['values'], compose([b,a])['weight']['values'])
    assert compose([]) == {}
    with pytest.raises(ValueError, match='shape'):
        compose([a, bank([0], [1], (3,2))])


def test_nested_disabled_and_exception_restores_base():
    net = nn.Sequential(nn.Linear(3,2,bias=False))
    original = net[0].weight.detach().clone()
    with pytest.raises(RuntimeError):
        with sparse_delta_supports(net, ['0.weight']) as mods:
            mods['0.weight'].delta.data.fill_(0.25)
            with disabled(mods):
                with disabled(mods):
                    assert torch.equal(net[0].weight, original)
                assert torch.equal(net[0].weight, original)
            assert not torch.equal(net[0].weight, original)
            raise RuntimeError('test exception')
    assert torch.equal(net[0].weight, original)
    assert not hasattr(net[0], 'parametrizations')


def test_frozen_library_gradient_and_modes():
    net = nn.Sequential(nn.Linear(3,2,bias=False))
    net.requires_grad_(False)
    original = net[0].weight.detach().clone()
    with sparse_delta_supports(net, ['0.weight']) as mods:
        p = bank([0], [0.5])
        install_library(mods, {'0.weight': p['weight']})
        frozen = mods['0.weight'].library.clone()
        set_mode(mods, 'union')
        net(torch.ones(1,3)).sum().backward()
        assert torch.count_nonzero(mods['0.weight'].delta.grad) == 6
        assert not mods['0.weight'].library.requires_grad
        assert torch.equal(mods['0.weight'].library, frozen)
        set_mode(mods, 'off')
        assert torch.equal(net[0].weight, original)
        with pytest.raises(ValueError):
            set_mode(mods, 'typo')
    assert torch.equal(net[0].weight, original)


def test_library_selection_order_round_once_and_restore(tmp_path):
    net = nn.Linear(3,2,bias=False).half()
    original = net.weight.detach().clone()
    a, b = bank([0,1],[0.1234,0.1]), bank([0,5],[-0.013,0.2])
    torch.save(a, tmp_path/'cat.pt')
    torch.save(b, tmp_path/'dog.pt')
    library = ModuleLibrary.from_directory(tmp_path)
    with library.activate(net, ['dog','cat']):
        first = net.weight.detach().clone()
        expected = original.float().clone().flatten()
        expected[0] += torch.tensor(0.1234)+torch.tensor(-0.013)
        expected[1] += 0.1
        expected[5] += 0.2
        assert torch.equal(first, expected.reshape_as(first).half())
    assert torch.equal(net.weight, original)
    with library.activate(net, ['cat','dog']):
        assert torch.equal(net.weight, first)
    with pytest.raises(RuntimeError):
        with library.activate(net, ['cat']):
            raise RuntimeError('test exception')
    assert torch.equal(net.weight, original)
    with library.activate(net, []):
        assert torch.equal(net.weight, original)
    with pytest.raises(ValueError):
        with library.activate(net, ['cat','cat']):
            pass
    with pytest.raises(ValueError):
        with library.activate(nn.Linear(2,2,bias=False), ['cat']):
            pass
