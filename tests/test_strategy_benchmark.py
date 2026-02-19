"""Tests for Strategy Selector benchmark."""

import pytest
import torch

from cortex_net.situation_encoder import SituationEncoder
from cortex_net.strategy_selector import StrategySelector, StrategyRegistry
from cortex_net.strategy_benchmark import (
    build_strategy_scenarios,
    train_strategy_selector,
    run_strategy_benchmark,
)


def test_scenarios_cover_all_strategies():
    scenarios = build_strategy_scenarios()
    registry = StrategyRegistry()
    covered = {s.best_strategy_id for s in scenarios}
    assert covered == set(registry.ids), f"Missing: {set(registry.ids) - covered}"


def test_trained_selector_beats_fixed():
    """The key test: learned selection outperforms fixed strategy."""
    torch.manual_seed(42)
    dim = 32

    encoder = SituationEncoder(text_dim=dim, output_dim=dim, hidden_dim=64, dropout=0.0)
    selector = StrategySelector(situation_dim=dim, num_strategies=10, hidden_dim=64)
    registry = StrategyRegistry()
    scenarios = build_strategy_scenarios()

    # Train
    losses = train_strategy_selector(encoder, selector, registry, scenarios, epochs=300, lr=1e-3)
    assert losses[-1] < losses[0]

    # Evaluate
    result = run_strategy_benchmark(encoder, selector, registry, scenarios)

    print(f"\n=== STRATEGY BENCHMARK ===")
    print(result.summary())
    print(f"\nLoss: {losses[0]:.4f} → {losses[-1]:.4f}")

    # Should beat fixed baseline
    assert result.accuracy > result.fixed_accuracy, (
        f"Learned ({result.accuracy:.1%}) should beat fixed ({result.fixed_accuracy:.1%})"
    )

    # Should use at least 5 distinct strategies
    assert result.num_strategies_used >= 5, (
        f"Only using {result.num_strategies_used} strategies, need ≥5"
    )
