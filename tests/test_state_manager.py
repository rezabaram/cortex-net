"""Tests for StateManager."""

import torch
import torch.nn as nn
import pytest

from cortex_net.state_manager import StateManager, CheckpointMetadata


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)


@pytest.fixture
def tmp_state_dir(tmp_path):
    return tmp_path / "state"


@pytest.fixture
def sm(tmp_state_dir):
    return StateManager(tmp_state_dir, max_checkpoints=3)


def test_save_and_load(sm):
    model = TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    meta = CheckpointMetadata(component_name="test", epoch=1, step=10)

    # Modify weights to something non-default
    with torch.no_grad():
        model.linear.weight.fill_(42.0)

    sm.save(model, opt, meta)

    # Load into a fresh model
    model2 = TinyModel()
    loaded_meta = sm.load(model2, component_name="test")

    assert loaded_meta is not None
    assert loaded_meta.epoch == 1
    assert loaded_meta.step == 10
    assert torch.allclose(model2.linear.weight, torch.full_like(model2.linear.weight, 42.0))


def test_load_no_checkpoint(sm):
    model = TinyModel()
    result = sm.load(model, component_name="nonexistent")
    assert result is None


def test_latest_checkpoint(sm):
    model = TinyModel()
    for step in [1, 5, 10]:
        sm.save(model, None, CheckpointMetadata(component_name="test", step=step))

    latest = sm.latest_checkpoint("test")
    assert latest is not None
    assert "00000010" in latest.name


def test_pruning(sm):
    model = TinyModel()
    for step in range(6):
        sm.save(model, None, CheckpointMetadata(component_name="test", step=step))

    checkpoints = sm.list_checkpoints("test")
    assert len(checkpoints) == 3  # max_checkpoints=3
    # Should keep the 3 most recent
    assert "00000003" in checkpoints[0].name


def test_atomic_write_no_corruption(sm, tmp_state_dir):
    """Verify no partial files left behind on success."""
    model = TinyModel()
    sm.save(model, None, CheckpointMetadata(component_name="test", step=1))

    component_dir = tmp_state_dir / "test"
    tmp_files = list(component_dir.glob("*.tmp"))
    assert len(tmp_files) == 0


def test_optimizer_state_roundtrip(sm):
    model = TinyModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    # Do a step so optimizer has state
    x = torch.randn(1, 4)
    loss = model.linear(x).sum()
    loss.backward()
    opt.step()

    sm.save(model, opt, CheckpointMetadata(component_name="test", step=1))

    model2 = TinyModel()
    opt2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    sm.load(model2, opt2, component_name="test")

    # Optimizer state should be populated
    assert len(opt2.state) > 0
