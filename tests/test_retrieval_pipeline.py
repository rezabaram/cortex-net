"""Tests for RetrievalPipeline."""

import pytest
import torch

from cortex_net.retrieval_pipeline import RetrievalPipeline, BenchmarkResult


@pytest.fixture
def pipeline(tmp_path):
    """Create a pipeline with small model for testing."""
    return RetrievalPipeline(
        state_dir=tmp_path / "state",
        log_dir=tmp_path / "logs",
        embedding_model="all-MiniLM-L6-v2",
        device="cpu",
    )


class TestRetrieval:
    def test_retrieve_returns_results(self, pipeline):
        candidates = [
            "The deployment process uses Docker containers",
            "Cats are fluffy animals",
            "CI/CD pipeline runs on GitHub Actions",
            "The weather is nice today",
            "Kubernetes orchestrates container deployments",
        ]
        result = pipeline.retrieve("How do we deploy?", candidates, k=3)

        assert len(result.indices) == 3
        assert len(result.scores) == 3
        assert len(result.texts) == 3
        assert result.interaction_id
        assert result.method in ("memory_gate", "cosine")

    def test_retrieve_ranks_relevant_higher(self, pipeline):
        candidates = [
            "Python is a programming language",
            "The capital of France is Paris",
            "Machine learning uses neural networks for pattern recognition",
            "Bananas are yellow fruit",
            "Deep learning is a subset of machine learning",
        ]
        result = pipeline.retrieve("Tell me about neural networks", candidates, k=2)

        # At least one ML-related candidate should be in top-2
        ml_indices = {2, 4}
        assert any(i in ml_indices for i in result.indices)


class TestOutcomeLogging:
    def test_log_outcome(self, pipeline):
        candidates = ["mem1", "mem2", "mem3"]
        result = pipeline.retrieve("query", candidates, k=2)
        pipeline.log_outcome(
            result.interaction_id,
            signal="positive",
            referenced=[0],
        )
        assert pipeline.logger.count() == 1

    def test_log_unknown_interaction_id(self, pipeline):
        # Should not crash
        pipeline.log_outcome("nonexistent", signal="positive")
        assert pipeline.logger.count() == 0


class TestCheckpointing:
    def test_save_and_load(self, pipeline):
        # Do a retrieve to change state
        pipeline.retrieve("test", ["a", "b"], k=1)
        pipeline.log_outcome(
            list(pipeline._pending.keys())[0] if pipeline._pending else "x",
            signal="positive",
            referenced=[0],
        )
        pipeline.save()

        # Create new pipeline, load state
        pipeline2 = RetrievalPipeline(
            state_dir=pipeline.state_manager.state_dir,
            log_dir=pipeline.logger.log_dir,
            device="cpu",
        )
        loaded = pipeline2.load()
        assert loaded

        # Weights should match
        assert torch.allclose(pipeline.gate.W, pipeline2.gate.W)

    def test_load_no_checkpoint(self, pipeline):
        assert not pipeline.load()


class TestTraining:
    def test_train_with_no_data(self, pipeline):
        metrics = pipeline.train(epochs=1)
        assert metrics.num_training_pairs == 0
        assert metrics.total_steps == 0

    def test_train_after_logging(self, pipeline):
        # Simulate some interactions with outcomes
        for i in range(5):
            result = pipeline.retrieve(
                f"query {i}",
                [f"relevant to query {i}", "irrelevant stuff", "random text"],
                k=2,
            )
            pipeline.log_outcome(
                result.interaction_id,
                signal="positive",
                referenced=[0],
            )

        metrics = pipeline.train(epochs=3)
        assert metrics.num_training_pairs > 0
        assert metrics.total_steps > 0
        assert metrics.avg_loss >= 0
        assert pipeline.gate.trained


class TestBenchmark:
    def test_benchmark_runs(self, pipeline):
        candidates = [
            "Docker containers for deployment",
            "Cats and dogs",
            "Kubernetes orchestration",
            "Weather forecast",
            "CI/CD pipelines",
        ]
        queries = ["How to deploy?", "Container orchestration?"]
        ground_truth = [{0, 2, 4}, {2}]

        result = pipeline.benchmark(queries, candidates, ground_truth, k=3)

        assert isinstance(result, BenchmarkResult)
        assert result.num_queries == 2
        assert result.k == 3
        assert 0 <= result.cosine_precision_at_k <= 1
        assert result.gate_wins + result.cosine_wins + result.ties == 2
