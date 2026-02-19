"""Tests for MemoryGate."""

import torch
import pytest

from cortex_net.memory_gate import MemoryGate
from cortex_net.state_manager import StateManager, CheckpointMetadata


@pytest.fixture
def gate():
    return MemoryGate(situation_dim=32, memory_dim=32)


def test_score_shape(gate):
    situation = torch.randn(32)
    memories = torch.randn(10, 32)
    scores = gate.score_memories(situation, memories)
    assert scores.shape == (10,)


def test_score_batch_shape(gate):
    situation = torch.randn(4, 32)
    memories = torch.randn(10, 32)
    scores = gate.score_memories(situation, memories)
    assert scores.shape == (4, 10)


def test_top_k_selection(gate):
    situation = torch.randn(32)
    memories = torch.randn(20, 32)
    indices, scores = gate.select_top_k(situation, memories, k=5)
    assert indices.shape == (5,)
    assert scores.shape == (5,)
    # Scores should be descending
    assert (scores[:-1] >= scores[1:]).all()


def test_top_k_clamps_to_n(gate):
    situation = torch.randn(32)
    memories = torch.randn(3, 32)
    indices, scores = gate.select_top_k(situation, memories, k=10)
    assert indices.shape == (3,)


def test_cosine_fallback_when_untrained(gate):
    assert not gate.trained
    situation = torch.randn(32)
    memories = torch.randn(5, 32)
    scores = gate.score_memories(situation, memories, use_cosine_fallback=True)
    # Cosine scores should be in [-1, 1]
    assert scores.min() >= -1.01
    assert scores.max() <= 1.01


def test_training_step(gate):
    optimizer = torch.optim.Adam(gate.parameters(), lr=0.01)
    situation = torch.randn(32)
    pos = torch.randn(3, 32)
    neg = torch.randn(7, 32)

    loss = gate.train_step(optimizer, situation, pos, neg)
    assert isinstance(loss, float)
    assert gate.trained


def test_training_improves_ranking():
    """After training, positive memories should rank higher."""
    torch.manual_seed(42)
    dim = 32
    gate = MemoryGate(situation_dim=dim, memory_dim=dim)
    optimizer = torch.optim.Adam(gate.parameters(), lr=0.01)

    situation = torch.randn(dim)
    pos_memories = torch.randn(3, dim) + situation * 0.1  # slight signal
    neg_memories = torch.randn(10, dim)

    # Train for a few steps
    for _ in range(50):
        gate.train_step(optimizer, situation, pos_memories, neg_memories)

    # Check: positive memories should score higher than negatives
    with torch.no_grad():
        pos_scores = gate.score_memories(situation, pos_memories, use_cosine_fallback=False)
        neg_scores = gate.score_memories(situation, neg_memories, use_cosine_fallback=False)
        assert pos_scores.mean() > neg_scores.mean()


def test_checkpoint_roundtrip(tmp_path):
    """MemoryGate saves and loads through StateManager."""
    gate = MemoryGate(situation_dim=16, memory_dim=16)
    optimizer = torch.optim.Adam(gate.parameters(), lr=0.01)

    # Train a bit
    for _ in range(10):
        gate.train_step(
            optimizer,
            torch.randn(16),
            torch.randn(2, 16),
            torch.randn(5, 16),
        )

    # Save
    sm = StateManager(tmp_path / "state")
    sm.save(gate, optimizer, CheckpointMetadata(component_name="memory_gate", step=10))

    # Load into fresh gate
    gate2 = MemoryGate(situation_dim=16, memory_dim=16)
    meta = sm.load(gate2, component_name="memory_gate")

    assert meta is not None
    assert torch.allclose(gate.W, gate2.W)
