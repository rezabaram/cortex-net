"""Tests for realistic benchmark."""

import pytest
import torch

from cortex_net.benchmark import (
    BenchmarkReport,
    build_scenarios,
    run_benchmark,
    train_on_scenarios,
)
from cortex_net.embedding_store import EmbeddingStore
from cortex_net.memory_gate import MemoryGate


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """Shared embedding store (expensive to init)."""
    p = tmp_path_factory.mktemp("store")
    return EmbeddingStore(p / "embeddings.jsonl")


@pytest.fixture
def gate():
    return MemoryGate(situation_dim=384, memory_dim=384)


def test_scenarios_are_valid():
    scenarios = build_scenarios()
    assert len(scenarios) >= 5
    for s in scenarios:
        assert s.query
        assert len(s.memories) >= 4
        assert s.relevant_indices
        assert all(i < len(s.memories) for i in s.relevant_indices)


def test_cosine_baseline(store, gate):
    """Run benchmark with untrained gate (= cosine fallback)."""
    report = run_benchmark(gate, store, k=3)
    assert report.num_scenarios >= 5
    # Cosine won't be perfect on these — that's the point
    print("\n=== COSINE BASELINE ===")
    print(report.summary())


def test_trained_gate_vs_cosine(store):
    """The key test: train the gate, then benchmark against cosine."""
    gate = MemoryGate(situation_dim=384, memory_dim=384)
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)

    # Baseline (untrained = cosine)
    baseline = run_benchmark(gate, store, k=3)

    # Train
    losses = train_on_scenarios(gate, optimizer, store, epochs=200, margin=1.0)
    assert losses[-1] < losses[0]  # loss decreased

    # Post-training
    trained = run_benchmark(gate, store, k=3)

    print("\n=== BEFORE TRAINING (cosine) ===")
    print(baseline.summary())
    print("\n=== AFTER TRAINING (learned gate) ===")
    print(trained.summary())
    print(f"\nLoss: {losses[0]:.4f} → {losses[-1]:.4f}")

    # The trained gate should improve over cosine
    assert trained.avg_gate_precision >= baseline.avg_cosine_precision, (
        f"Trained gate ({trained.avg_gate_precision:.3f}) should beat "
        f"cosine baseline ({baseline.avg_cosine_precision:.3f})"
    )


def test_gate_checkpoint_preserves_benchmark(store, tmp_path):
    """Train, save, load, verify benchmark results are identical."""
    from cortex_net.state_manager import StateManager, CheckpointMetadata

    gate = MemoryGate(situation_dim=384, memory_dim=384)
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)
    train_on_scenarios(gate, optimizer, store, epochs=100)

    report_before = run_benchmark(gate, store, k=3)

    # Save and load
    sm = StateManager(tmp_path / "state")
    sm.save(gate, optimizer, CheckpointMetadata(component_name="memory_gate", step=100))

    gate2 = MemoryGate(situation_dim=384, memory_dim=384)
    sm.load(gate2, component_name="memory_gate")

    report_after = run_benchmark(gate2, store, k=3)

    # Results should be identical
    for r1, r2 in zip(report_before.results, report_after.results):
        assert r1.gate_ranking == r2.gate_ranking
        assert r1.gate_precision == r2.gate_precision
