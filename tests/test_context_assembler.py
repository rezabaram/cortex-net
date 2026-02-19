"""Tests for Context Assembler — the full pipeline."""

import pytest
import torch

from cortex_net.context_assembler import ContextAssembler, AssembledContext


@pytest.fixture(scope="module")
def assembler(tmp_path_factory):
    state_dir = tmp_path_factory.mktemp("state")
    return ContextAssembler(
        text_dim=384,
        num_strategies=10,
        state_dir=state_dir,
        device="cpu",
    )


class TestAssembly:
    def test_assemble_returns_context(self, assembler):
        result = assembler.assemble(
            query="How do I deploy the service?",
            candidate_memories=[
                "Use docker-compose up in production.",
                "The CI pipeline handles deployments.",
                "Cats are fluffy animals.",
                "Check the runbook at /docs/deploy.md.",
            ],
            k=2,
        )
        assert isinstance(result, AssembledContext)
        assert len(result.memories) == 2
        assert len(result.memory_scores) == 2
        assert result.strategy is not None
        assert result.confidence is not None
        assert result.confidence.action in ("proceed", "hedge", "escalate")

    def test_assemble_with_history_and_metadata(self, assembler):
        result = assembler.assemble(
            query="What's the status?",
            candidate_memories=["Sprint is on track.", "Server is down.", "Lunch at noon."],
            k=2,
            history=["Let's do the standup", "Sure, go ahead"],
            metadata={"hour_of_day": 9, "day_of_week": 1, "is_group_chat": 1},
        )
        assert len(result.memories) == 2
        assert result.strategy.strategy_id in assembler.strategy_registry.ids

    def test_prompt_prefix_generated(self, assembler):
        result = assembler.assemble(
            query="Explain caching",
            candidate_memories=["Redis is used for caching.", "Memcached is an alternative."],
            k=2,
        )
        prefix = result.prompt_prefix
        assert isinstance(prefix, str)
        assert len(prefix) > 0

    def test_different_queries_different_results(self, assembler):
        r1 = assembler.assemble("Deploy the app", ["deploy guide", "test guide"], k=1)
        r2 = assembler.assemble("Write a poem", ["deploy guide", "test guide"], k=1)
        # Strategy or memories may differ
        # At minimum, situation embeddings should differ
        assert r1.situation_embedding != r2.situation_embedding


class TestStatePersistence:
    def test_save_and_load(self, assembler, tmp_path):
        assembler2 = ContextAssembler(
            text_dim=384, state_dir=tmp_path / "state2",
        )
        assembler2.save()

        assembler3 = ContextAssembler(
            text_dim=384, state_dir=tmp_path / "state2",
        )
        loaded = assembler3.load()
        assert loaded


class TestParameterCount:
    def test_parameter_count(self, assembler):
        counts = assembler.parameter_count()
        assert counts["total"] > 0
        assert counts["situation_encoder"] > 0
        assert counts["memory_gate"] > 0
        assert counts["strategy_selector"] > 0
        assert counts["confidence_estimator"] > 0
        # Should be small (< 10M as per design)
        assert counts["total"] < 10_000_000
        print(f"\nParameter counts: {counts}")
