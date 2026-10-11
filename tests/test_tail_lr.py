from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from loop_ultrasound import tail_lr
from loop_ultrasound.train import CaseViewSampler


def test_workers_partition_ten_prefixes_without_duplicate_tails():
    expected = {(s, d) for s in (53, 71, 89, 107, 131) for d in (2, 4)}
    a, b = set(tail_lr.assigned_pairs(2, 0)), set(tail_lr.assigned_pairs(2, 1))
    assert not a & b and a | b == expected and len(a) == len(b) == 5
    assert len(expected) * (2560 + 2 * 640) == 38400


@pytest.mark.parametrize("workers,index", [(0, 0), (3, 0), (2, 2), (1, 1), (2, -1)])
def test_invalid_workers_refused(workers, index):
    with pytest.raises(ValueError): tail_lr.assigned_pairs(workers, index)


def fixture_state():
    torch.manual_seed(53); np.random.seed(53); random.seed(53)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    rows = [{"case_id": str(i // 2)} for i in range(8)]
    dataset = TensorDataset(torch.arange(24, dtype=torch.float32).reshape(8, 3))
    loader = DataLoader(dataset, batch_size=2, sampler=CaseViewSampler(rows, 53),
                        generator=torch.Generator().manual_seed(53))
    for batch, in loader:
        optimizer.zero_grad(set_to_none=True)
        model(batch).square().mean().backward(); optimizer.step()
    loader.sampler.epoch = 20
    model.eval()
    return model, optimizer, loader, tail_lr.capture_state(model, optimizer, loader)


def test_disk_checkpoint_roundtrip_preserves_full_state_and_next_randomness(tmp_path):
    model, optimizer, loader, state = fixture_state()
    digest = tail_lr.state_digest(state)
    path = tmp_path / "checkpoint.pt"
    torch.save(state, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert tail_lr.state_digest(loaded) == digest
    random_before = (torch.rand(5), np.random.rand(5), [random.random() for _ in range(5)])
    batches_before = [x.clone() for x, in loader]
    tail_lr.restore_state(loaded, model, optimizer, loader)
    assert tail_lr.state_digest(tail_lr.capture_state(model, optimizer, loader)) == digest
    assert loader.sampler.epoch == 20 and model.training is False
    assert torch.equal(torch.rand(5), random_before[0])
    assert np.array_equal(np.random.rand(5), random_before[1])
    assert [random.random() for _ in range(5)] == random_before[2]
    assert all(torch.equal(a, b) for a, b in zip(batches_before, [x for x, in loader]))


def test_one_branch_cannot_mutate_the_prefix_or_other_branch_optimizer():
    model, optimizer, loader, state = fixture_state()
    original = tail_lr.state_digest(state)
    tail_lr.restore_state(state, model, optimizer, loader)
    for group in optimizer.param_groups: group['lr'] = .00003
    for batch, in loader:
        optimizer.zero_grad(set_to_none=True)
        model(batch).square().mean().backward(); optimizer.step()
    assert tail_lr.state_digest(state) == original
    tail_lr.restore_state(state, model, optimizer, loader)
    assert optimizer.param_groups[0]['lr'] == .0003
    assert tail_lr.state_digest(tail_lr.capture_state(model, optimizer, loader)) == original


def test_restored_sampler_seed_must_match():
    model, optimizer, loader, state = fixture_state()
    state['sampler']['seed'] = 71
    with pytest.raises(ValueError, match="sampler seed"):
        tail_lr.restore_state(state, model, optimizer, loader)


@pytest.mark.parametrize("part", ['model_state', 'optimizer_state', 'loader_generator_state', 'sampler'])
def test_any_changed_full_state_component_changes_digest(part):
    _, _, _, state = fixture_state()
    altered = deepcopy(state)
    if part == 'model_state': altered[part]['weight'][0, 0] += 1
    elif part == 'optimizer_state': altered[part]['param_groups'][0]['lr'] = .00003
    elif part == 'loader_generator_state': altered[part][0] = 71
    else: altered[part]['epoch'] = 21
    assert tail_lr.state_digest(state) != tail_lr.state_digest(altered)


def test_policy_change_is_only_after_prefix_and_keeps_budget():
    constant = tail_lr.branch_config(53, 4, 'constant')
    low = tail_lr.branch_config(53, 4, 'low_tail')
    assert constant['learning_rate'] == low['learning_rate'] == .0003
    assert low['tail_learning_rate'] == .00003 and constant['tail_learning_rate'] == .0003
    assert constant['prefix_updates'] == low['prefix_updates'] == 2560
    assert constant['epochs'] == low['epochs'] == 25
