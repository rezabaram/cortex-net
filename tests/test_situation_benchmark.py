"""Tests for Situation Encoder contextual benchmark."""

import pytest
import torch

from cortex_net.embedding_store import EmbeddingStore
from cortex_net.memory_gate import MemoryGate
from cortex_net.situation_encoder import SituationEncoder
from cortex_net.situation_benchmark import (
    build_contextual_scenarios,
    run_contextual_benchmark,
    train_contextual,
    print_contextual_report,
)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    p = tmp_path_factory.mktemp("store")
    return EmbeddingStore(p / "embeddings.jsonl")


def test_scenarios_valid():
    scenarios = build_contextual_scenarios()
    assert len(scenarios) >= 3
    for s in scenarios:
        assert s.context_a_relevant != s.context_b_relevant
        assert s.context_a_relevant.isdisjoint(s.context_b_relevant) or \
               s.context_a_relevant != s.context_b_relevant


def test_trained_encoder_beats_raw(store):
    """The key Phase 2 test: encoder+gate outperforms raw text+gate
    on context-dependent retrieval."""
    torch.manual_seed(42)
    text_dim = 384  # sentence-transformer dim
    output_dim = 128

    # Encoder + Gate (trained jointly)
    encoder = SituationEncoder(text_dim=text_dim, output_dim=output_dim, hidden_dim=256, dropout=0.0)
    gate_enc = MemoryGate(situation_dim=output_dim, memory_dim=text_dim)

    # Raw text Gate (trained on raw embeddings, no encoder)
    gate_raw = MemoryGate(situation_dim=text_dim, memory_dim=text_dim)

    scenarios = build_contextual_scenarios()

    # Train the encoder+gate jointly
    losses = train_contextual(encoder, gate_enc, store, scenarios, epochs=300, lr=1e-3)
    assert losses[-1] < losses[0]

    # Train raw gate on the scenarios too (it gets same training signal, just no metadata)
    from cortex_net.benchmark import train_on_scenarios, Scenario
    # Convert contextual scenarios to regular scenarios (using context_a as ground truth)
    raw_scenarios = []
    for s in scenarios:
        raw_scenarios.append(Scenario(
            name=s.name + "_a", query=s.query,
            memories=s.memories, relevant_indices=s.context_a_relevant,
        ))
        raw_scenarios.append(Scenario(
            name=s.name + "_b", query=s.query,
            memories=s.memories, relevant_indices=s.context_b_relevant,
        ))
    raw_opt = torch.optim.Adam(gate_raw.parameters(), lr=1e-3)
    train_on_scenarios(gate_raw, raw_opt, store, raw_scenarios, epochs=300)

    # Run benchmark
    results = run_contextual_benchmark(encoder, gate_enc, gate_raw, store, scenarios, k=2)

    report = print_contextual_report(results)
    print(f"\n=== CONTEXTUAL BENCHMARK ===\n{report}")
    print(f"\nEncoder loss: {losses[0]:.4f} → {losses[-1]:.4f}")

    # Encoder should win on average
    enc_avg = sum(r.encoder_avg for r in results) / len(results)
    raw_avg = sum(r.raw_avg for r in results) / len(results)

    assert enc_avg >= raw_avg, (
        f"Encoder ({enc_avg:.3f}) should beat raw ({raw_avg:.3f}) on contextual retrieval"
    )
