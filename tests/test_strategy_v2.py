"""Tests for the redesigned strategy system — sets + continuous selector."""

import pytest
import torch

from cortex_net.strategy_selector import (
    StrategyProfile,
    StrategyRegistry,
    StrategySelector,
    ContinuousStrategySelector,
    ContinuousStrategyOutput,
    get_strategy_set,
    list_strategy_sets,
    merge_strategy_sets,
    STRATEGY_SETS,
    DEFAULT_STRATEGIES,
)


class TestStrategySets:
    def test_list_sets(self):
        sets = list_strategy_sets()
        assert "generic" in sets
        assert "developer" in sets
        assert "support" in sets

    def test_get_set(self):
        dev = get_strategy_set("developer")
        assert len(dev) == 12
        ids = [s.id for s in dev]
        assert "implement" in ids
        assert "debug" in ids
        assert "refactor" in ids
        assert "review" in ids
        assert "test" in ids
        assert "explain" in ids
        assert "architect" in ids
        assert "quick_fix" in ids
        assert "deploy" in ids

    def test_get_set_returns_copy(self):
        s1 = get_strategy_set("developer")
        s2 = get_strategy_set("developer")
        assert s1 is not s2

    def test_get_unknown_set(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_strategy_set("nonexistent")

    def test_support_set(self):
        sup = get_strategy_set("support")
        ids = [s.id for s in sup]
        assert "diagnose" in ids
        assert "escalate" in ids
        assert "empathize" in ids

    def test_merge_sets(self):
        merged = merge_strategy_sets("generic", "developer")
        ids = [s.id for s in merged]
        # Should have both generic-only and developer-only strategies
        assert "creative" in ids  # generic only
        assert "implement" in ids  # developer only

    def test_merge_dedup(self):
        # Both generic and support have "empathize"
        merged = merge_strategy_sets("generic", "support")
        empathize_count = sum(1 for s in merged if s.id == "empathize")
        assert empathize_count == 1

    def test_generic_backward_compat(self):
        assert DEFAULT_STRATEGIES == STRATEGY_SETS["generic"]


class TestRegistryFromSet:
    def test_from_set(self):
        reg = StrategyRegistry.from_set("developer")
        assert len(reg) == 12
        assert reg.get_by_id("implement") is not None

    def test_from_sets(self):
        reg = StrategyRegistry.from_sets("generic", "developer")
        assert reg.get_by_id("creative") is not None
        assert reg.get_by_id("implement") is not None

    def test_add_dynamic(self):
        reg = StrategyRegistry.from_set("developer")
        n = len(reg)
        reg.add(StrategyProfile(id="custom", name="Custom", description="Custom strategy"))
        assert len(reg) == n + 1
        assert reg.get_by_id("custom") is not None

    def test_add_replaces_existing(self):
        reg = StrategyRegistry.from_set("developer")
        n = len(reg)
        reg.add(StrategyProfile(id="debug", name="New Debug", description="Replaced"))
        assert len(reg) == n  # no new entry
        assert reg.get_by_id("debug").name == "New Debug"


class TestContinuousSelector:
    @pytest.fixture
    def selector(self):
        return ContinuousStrategySelector(
            situation_dim=384,
            strategy_dim=64,
            num_anchors=12,
        )

    @pytest.fixture
    def registry(self):
        return StrategyRegistry.from_set("developer")

    def test_forward_shape(self, selector):
        x = torch.randn(384)
        out = selector(x)
        assert out.shape == (64,)
        # L2 normalized
        assert abs(out.norm().item() - 1.0) < 1e-5

    def test_forward_batch(self, selector):
        x = torch.randn(4, 384)
        out = selector(x)
        assert out.shape == (4, 64)

    def test_init_anchors(self, selector, registry):
        selector.init_anchors_from_registry(registry)
        assert len(selector.anchor_ids) == 12

    def test_select(self, selector, registry):
        selector.init_anchors_from_registry(registry)
        x = torch.randn(384)
        result = selector.select(x, registry)
        assert isinstance(result, ContinuousStrategyOutput)
        assert result.embedding.shape == (64,)
        assert len(result.nearest_strategies) <= 3
        assert len(result.weights) > 0
        assert "reasoning_depth" in result.attributes
        assert "creativity" in result.attributes

    def test_select_no_anchors(self, selector, registry):
        x = torch.randn(384)
        result = selector.select(x, registry)
        assert result.nearest_strategies == []

    def test_primary(self, selector, registry):
        selector.init_anchors_from_registry(registry)
        x = torch.randn(384)
        result = selector.select(x, registry)
        assert result.primary in registry.ids

    def test_prompt_framing(self, selector, registry):
        selector.init_anchors_from_registry(registry)
        x = torch.randn(384)
        result = selector.select(x, registry)
        framing = result.prompt_framing(registry)
        assert isinstance(framing, str)

    def test_anchor_loss(self, selector, registry):
        selector.init_anchors_from_registry(registry)
        x = torch.randn(384)
        loss = selector.anchor_loss(x, 0)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_attribute_loss(self, selector, registry):
        selector.init_anchors_from_registry(registry)
        x = torch.randn(384)
        loss = selector.attribute_loss(x, {"reasoning_depth": 0.8, "creativity": 0.2})
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_attributes_in_range(self, selector, registry):
        selector.init_anchors_from_registry(registry)
        x = torch.randn(384)
        result = selector.select(x, registry)
        for name, val in result.attributes.items():
            assert 0.0 <= val <= 1.0, f"{name}={val} out of [0,1]"

    def test_trainable(self, selector, registry):
        selector.init_anchors_from_registry(registry)
        opt = torch.optim.Adam(selector.parameters(), lr=1e-3)
        x = torch.randn(384)

        loss_before = selector.anchor_loss(x, 0).item()
        for _ in range(20):
            opt.zero_grad()
            loss = selector.anchor_loss(x, 0)
            loss.backward()
            opt.step()
        loss_after = selector.anchor_loss(x, 0).item()
        # Should decrease or stay near 0
        assert loss_after <= loss_before + 0.01


class TestCategoricalBackwardCompat:
    """Ensure old StrategySelector still works unchanged."""

    def test_original_selector(self):
        sel = StrategySelector(situation_dim=384, num_strategies=7)
        reg = StrategyRegistry.from_set("generic")
        x = torch.randn(384)
        result = sel.select(x, reg)
        assert result.strategy_id in reg.ids

    def test_developer_categorical(self):
        reg = StrategyRegistry.from_set("developer")
        sel = StrategySelector(situation_dim=384, num_strategies=len(reg))
        x = torch.randn(384)
        result = sel.select(x, reg)
        assert result.strategy_id in reg.ids
