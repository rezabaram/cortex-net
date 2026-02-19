"""Tests for Strategy Selector."""

import pytest
import torch

from cortex_net.strategy_selector import (
    StrategySelector,
    StrategyRegistry,
    StrategyProfile,
    DEFAULT_STRATEGIES,
)
from cortex_net.state_manager import StateManager, CheckpointMetadata


@pytest.fixture
def registry():
    return StrategyRegistry()  # generic set


@pytest.fixture
def selector(registry):
    return StrategySelector(situation_dim=32, num_strategies=len(registry), hidden_dim=64)


class TestStrategyRegistry:
    def test_default_strategies(self, registry):
        assert len(registry) == 7
        assert registry[0].id == "deep_research"
        assert registry.get_by_id("quick_answer") is not None

    def test_custom_registry(self):
        custom = StrategyRegistry([
            StrategyProfile(id="a", name="A", description="Strategy A"),
            StrategyProfile(id="b", name="B", description="Strategy B"),
        ])
        assert len(custom) == 2
        assert custom.id_to_index("b") == 1

    def test_unknown_id(self, registry):
        assert registry.get_by_id("nonexistent") is None


class TestStrategySelector:
    def test_forward_shape(self, selector, registry):
        sit = torch.randn(32)
        logits = selector(sit)
        assert logits.shape == (len(registry),)

    def test_forward_batch(self, selector, registry):
        sit = torch.randn(4, 32)
        logits = selector(sit)
        assert logits.shape == (4, len(registry))

    def test_select_returns_valid(self, selector, registry):
        sit = torch.randn(32)
        selection = selector.select(sit, registry, explore=False)
        assert selection.strategy_id in registry.ids
        assert 0 <= selection.confidence <= 1
        assert abs(sum(selection.probabilities.values()) - 1.0) < 0.01

    def test_exploration_varies_choices(self, selector, registry):
        """With high exploration, selections should vary."""
        selector.exploration_rate = 1.0  # always explore
        sit = torch.randn(32)
        strategies = set()
        for _ in range(50):
            sel = selector.select(sit, registry, explore=True)
            strategies.add(sel.strategy_id)
        # Should pick at least 3 different strategies with random exploration
        assert len(strategies) >= 3

    def test_no_exploration_deterministic(self, selector, registry):
        """Without exploration, same input → same output."""
        selector.exploration_rate = 0.0
        sit = torch.randn(32)
        selections = [selector.select(sit, registry, explore=False).strategy_id for _ in range(10)]
        assert len(set(selections)) == 1


class TestTraining:
    def test_training_step(self, selector):
        optimizer = torch.optim.Adam(selector.parameters(), lr=0.01)
        sit = torch.randn(32)
        loss = selector.train_step(optimizer, sit, target_strategy_idx=3)
        assert isinstance(loss, float)
        assert loss > 0

    def test_training_converges(self):
        """Selector should learn to map specific situations to specific strategies."""
        torch.manual_seed(42)
        selector = StrategySelector(situation_dim=16, num_strategies=5, hidden_dim=32)
        optimizer = torch.optim.Adam(selector.parameters(), lr=0.01)
        registry = StrategyRegistry(DEFAULT_STRATEGIES[:5])

        # Create 5 distinct situation vectors, one per strategy
        situations = [torch.randn(16) for _ in range(5)]

        losses = []
        for epoch in range(100):
            epoch_loss = 0
            for i, sit in enumerate(situations):
                loss = selector.train_step(optimizer, sit, target_strategy_idx=i)
                epoch_loss += loss
            losses.append(epoch_loss / 5)

        # Loss should decrease
        assert losses[-1] < losses[0]

        # Should correctly classify all 5 situations
        correct = 0
        for i, sit in enumerate(situations):
            sel = selector.select(sit, registry, explore=False)
            if registry.id_to_index(sel.strategy_id) == i:
                correct += 1
        assert correct >= 4  # at least 4 out of 5

    def test_diversity_score(self):
        selector = StrategySelector(situation_dim=16, num_strategies=4)
        # No usage yet
        assert selector.diversity_score() == 0.0

        # Uniform usage
        selector._usage_counts = torch.tensor([10, 10, 10, 10])
        assert selector.diversity_score() == pytest.approx(1.0, abs=0.01)

        # Single strategy only
        selector._usage_counts = torch.tensor([100, 0, 0, 0])
        assert selector.diversity_score() == 0.0


class TestCheckpointing:
    def test_checkpoint_roundtrip(self, tmp_path):
        selector = StrategySelector(situation_dim=16, num_strategies=5)
        optimizer = torch.optim.Adam(selector.parameters(), lr=0.01)

        # Train a bit
        for _ in range(10):
            selector.train_step(optimizer, torch.randn(16), target_strategy_idx=2)

        sm = StateManager(tmp_path / "state")
        sm.save(selector, optimizer, CheckpointMetadata(component_name="strategy_selector", step=10))

        selector2 = StrategySelector(situation_dim=16, num_strategies=5)
        meta = sm.load(selector2, component_name="strategy_selector")

        assert meta is not None
        # Weights should match
        for p1, p2 in zip(selector.parameters(), selector2.parameters()):
            assert torch.allclose(p1, p2)


class TestIntegration:
    def test_with_situation_encoder(self):
        """Situation Encoder → Strategy Selector pipeline."""
        from cortex_net.situation_encoder import SituationEncoder, extract_metadata_features

        encoder = SituationEncoder(text_dim=32, output_dim=64, hidden_dim=128)
        selector = StrategySelector(situation_dim=64, num_strategies=10)
        registry = StrategyRegistry()

        # Encode a situation
        msg = torch.randn(32)
        hist = torch.randn(32)
        meta = extract_metadata_features({"hour_of_day": 14})

        with torch.no_grad():
            sit = encoder(msg, hist, meta)
            selection = selector.select(sit, registry, explore=False)

        assert selection.strategy_id in registry.ids
        assert selection.strategy is not None
