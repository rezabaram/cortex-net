"""Tests for evaluation harness."""

import torch
import pytest

from cortex_net.eval import (
    precision_at_k,
    recall_at_k,
    evaluate_retrieval,
    generate_synthetic_data,
)
from cortex_net.memory_gate import MemoryGate


def test_precision_at_k():
    retrieved = torch.tensor([0, 1, 2, 3, 4])
    relevant = {0, 2, 4}
    assert precision_at_k(retrieved, relevant, k=5) == 3 / 5
    assert precision_at_k(retrieved, relevant, k=1) == 1.0
    assert precision_at_k(retrieved, relevant, k=2) == 0.5


def test_recall_at_k():
    retrieved = torch.tensor([0, 1, 2, 3, 4])
    relevant = {0, 2, 7}
    assert recall_at_k(retrieved, relevant, k=5) == 2 / 3
    assert recall_at_k(retrieved, relevant, k=1) == 1 / 3


def test_recall_empty_relevant():
    retrieved = torch.tensor([0, 1])
    assert recall_at_k(retrieved, set(), k=2) == 0.0


def test_generate_synthetic_data():
    situation, memories, relevant = generate_synthetic_data(
        num_memories=50, num_relevant=5, dim=64
    )
    assert situation.shape == (64,)
    assert memories.shape == (50, 64)
    assert len(relevant) == 5


def test_evaluate_retrieval_runs():
    situation, memories, relevant = generate_synthetic_data(
        num_memories=30, num_relevant=3, dim=32
    )
    gate = MemoryGate(situation_dim=32, memory_dim=32)
    gate_m, cosine_m = evaluate_retrieval(gate, situation, memories, relevant, k=5)

    assert 0 <= gate_m.precision_at_k <= 1
    assert 0 <= cosine_m.precision_at_k <= 1
    assert gate_m.k == 5


def test_trained_gate_beats_cosine_on_hard_data():
    """With low signal strength, trained gate should outperform cosine."""
    torch.manual_seed(123)
    dim = 64
    situation, memories, relevant = generate_synthetic_data(
        num_memories=50, num_relevant=5, dim=dim, signal_strength=0.1
    )

    gate = MemoryGate(situation_dim=dim, memory_dim=dim)
    optimizer = torch.optim.Adam(gate.parameters(), lr=0.01)

    # Train on this data
    pos = memories[list(relevant)]
    neg_idx = [i for i in range(50) if i not in relevant]
    neg = memories[neg_idx]

    for _ in range(100):
        gate.train_step(optimizer, situation, pos, neg, margin=1.0)

    gate_m, cosine_m = evaluate_retrieval(gate, situation, memories, relevant, k=5)
    # Trained gate should achieve perfect or near-perfect precision on training data
    assert gate_m.precision_at_k >= cosine_m.precision_at_k
