"""Tests for comparison benchmark."""

import pytest
import torch

from cortex_net.comparison_benchmark import (
    build_comparison_scenarios,
    run_comparison,
)


def test_scenarios_have_valid_structure():
    scenarios = build_comparison_scenarios()
    assert len(scenarios) >= 5
    for s in scenarios:
        assert s.query
        assert len(s.memories) >= 3
        assert s.relevant_indices
        assert all(i < len(s.memories) for i in s.relevant_indices)
        assert s.expected_strategy
        assert s.expected_confidence in ("proceed", "hedge", "escalate")


@pytest.mark.slow
def test_cortex_net_beats_baselines():
    """The key test: cortex-net outperforms cosine RAG and no-memory."""
    result = run_comparison(train_epochs=150)

    print(f"\n{result.summary()}")

    # cortex-net should beat cosine RAG on precision
    assert result.cortex_net.mean_precision >= result.cosine_rag.mean_precision, (
        f"cortex-net P@3 ({result.cortex_net.mean_precision:.3f}) should beat "
        f"cosine RAG ({result.cosine_rag.mean_precision:.3f})"
    )

    # cortex-net should have non-zero strategy accuracy
    assert result.cortex_net.strategy_accuracy > 0.3, (
        f"Strategy accuracy {result.cortex_net.strategy_accuracy:.1%} too low"
    )

    # cortex-net should have non-zero confidence alignment
    assert result.cortex_net.confidence_alignment > 0.3, (
        f"Confidence alignment {result.cortex_net.confidence_alignment:.1%} too low"
    )

    # Obviously beat no-memory
    assert result.cortex_net.mean_precision > result.no_memory.mean_precision
