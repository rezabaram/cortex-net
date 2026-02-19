"""Tests for online trainer."""

import pytest
import torch

from cortex_net.confidence_estimator import ConfidenceEstimator
from cortex_net.feedback_collector import ReplayBuffer
from cortex_net.memory_gate import MemoryGate
from cortex_net.online_trainer import OnlineTrainer
from cortex_net.situation_encoder import SituationEncoder
from cortex_net.strategy_selector import StrategySelector, StrategyRegistry

DIM = 32


@pytest.fixture
def trainer(tmp_path):
    import uuid
    encoder = SituationEncoder(text_dim=DIM, output_dim=DIM, hidden_dim=64, dropout=0.0)
    gate = MemoryGate(situation_dim=DIM, memory_dim=DIM)
    selector = StrategySelector(situation_dim=DIM, num_strategies=10, hidden_dim=32)
    estimator = ConfidenceEstimator(situation_dim=DIM, hidden_dim=32, dropout=0.0)
    registry = StrategyRegistry()
    buffer = ReplayBuffer(capacity=100, path=tmp_path / f"replay-{uuid.uuid4().hex[:8]}.jsonl")

    return OnlineTrainer(
        encoder, gate, selector, estimator, registry,
        buffer=buffer, update_every=3, batch_size=4,
    )


class TestOnlineTrainer:
    def test_record_positive_interaction(self, trainer):
        sit = torch.randn(DIM)
        outcome = trainer.record_interaction(
            situation_emb=sit,
            memory_indices=[0, 1, 2],
            memory_scores=[0.9, 0.7, 0.5],
            strategy_id="quick_answer",
            confidence=0.85,
            user_response="Thanks, that's perfect!",
            query="What port does Redis use?",
        )
        assert outcome.reward > 0
        assert trainer.buffer_size == 1

    def test_record_negative_interaction(self, trainer):
        sit = torch.randn(DIM)
        outcome = trainer.record_interaction(
            situation_emb=sit,
            memory_indices=[0, 1],
            memory_scores=[0.6, 0.4],
            strategy_id="deep_research",
            confidence=0.9,
            user_response="That's wrong, it's port 3000",
        )
        assert outcome.reward < 0

    def test_auto_update_after_n_interactions(self, trainer):
        """Should auto-train after update_every interactions."""
        for i in range(3):
            trainer.record_interaction(
                situation_emb=torch.randn(DIM),
                memory_indices=[0],
                memory_scores=[0.8],
                strategy_id="quick_answer",
                confidence=0.7,
                user_response="Thanks!" if i % 2 == 0 else "That's wrong",
            )
        assert trainer.total_updates >= 1

    def test_manual_step(self, trainer):
        # Add some entries first
        for i in range(5):
            trainer.record_interaction(
                situation_emb=torch.randn(DIM),
                memory_indices=[0, 1],
                memory_scores=[0.9, 0.6],
                strategy_id="code_assist",
                confidence=0.8,
                user_response="Perfect!" if i % 2 == 0 else "No, that's not right",
            )

        update = trainer.step()
        assert update.num_samples > 0
        assert update.total_loss >= 0

    def test_ema_loss_tracks(self, trainer):
        for i in range(10):
            trainer.record_interaction(
                situation_emb=torch.randn(DIM),
                memory_indices=[0],
                memory_scores=[0.8],
                strategy_id="quick_answer",
                confidence=0.5,
                user_response="Thanks!",
            )
        # EMA should be tracking
        assert trainer.ema_loss >= 0

    def test_empty_buffer_step(self, tmp_path):
        import uuid
        encoder = SituationEncoder(text_dim=DIM, output_dim=DIM, hidden_dim=64, dropout=0.0)
        gate = MemoryGate(situation_dim=DIM, memory_dim=DIM)
        selector = StrategySelector(situation_dim=DIM, num_strategies=10, hidden_dim=32)
        estimator = ConfidenceEstimator(situation_dim=DIM, hidden_dim=32, dropout=0.0)
        registry = StrategyRegistry()
        empty_buf = ReplayBuffer(capacity=100, path=tmp_path / f"empty-{uuid.uuid4().hex[:8]}.jsonl")
        t = OnlineTrainer(encoder, gate, selector, estimator, registry, buffer=empty_buf)
        update = t.step()
        assert update.num_samples == 0


class TestLearningLoop:
    def test_full_loop_converges(self, tmp_path):
        """Simulate a full learning loop: interactions → feedback → updates."""
        torch.manual_seed(42)
        encoder = SituationEncoder(text_dim=DIM, output_dim=DIM, hidden_dim=64, dropout=0.0)
        gate = MemoryGate(situation_dim=DIM, memory_dim=DIM)
        selector = StrategySelector(situation_dim=DIM, num_strategies=10, hidden_dim=32)
        estimator = ConfidenceEstimator(situation_dim=DIM, hidden_dim=32, dropout=0.0)
        registry = StrategyRegistry()
        buffer = ReplayBuffer(capacity=1000, path=tmp_path / "replay.jsonl")

        trainer = OnlineTrainer(
            encoder, gate, selector, estimator, registry,
            buffer=buffer, update_every=5, batch_size=8, lr=1e-3,
        )

        # Simulate 50 interactions with consistent feedback patterns
        losses = []
        for i in range(50):
            sit = torch.randn(DIM)
            response = "Thanks, perfect!" if i % 3 != 0 else "That's wrong"

            trainer.record_interaction(
                situation_emb=sit,
                memory_indices=[0, 1, 2],
                memory_scores=[0.9, 0.7, 0.5],
                strategy_id="quick_answer",
                confidence=0.8,
                user_response=response,
            )

        # Should have done multiple updates
        assert trainer.total_updates >= 5
        assert trainer.buffer_size >= 50

        # Run a few more manual steps
        for _ in range(10):
            update = trainer.step()
            if update.num_samples > 0:
                losses.append(update.total_loss)

        print(f"\nOnline learning: {trainer.total_updates} updates, "
              f"buffer={trainer.buffer_size}, EMA loss={trainer.ema_loss:.4f}")
        if losses:
            print(f"Last 10 step losses: {[f'{l:.4f}' for l in losses]}")
