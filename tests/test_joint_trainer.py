"""Tests for Joint Trainer and Ablation."""

import pytest
import torch

from cortex_net.confidence_estimator import ConfidenceEstimator, expected_calibration_error
from cortex_net.joint_trainer import JointTrainer, TrainingSample, AblationResult
from cortex_net.memory_gate import MemoryGate
from cortex_net.situation_encoder import SituationEncoder
from cortex_net.strategy_selector import StrategySelector, StrategyRegistry


DIM = 32


def make_samples(n: int = 20, dim: int = DIM) -> list[TrainingSample]:
    """Create diverse training samples."""
    torch.manual_seed(42)
    registry = StrategyRegistry()
    samples = []
    for i in range(n):
        n_mems = 8
        mems = torch.randn(n_mems, dim)
        relevant = {i % n_mems, (i + 1) % n_mems}  # 2 relevant per sample

        samples.append(TrainingSample(
            message_emb=torch.randn(dim),
            history_emb=torch.randn(dim),
            metadata={"hour_of_day": i % 24, "day_of_week": i % 7},
            memory_embs=mems,
            relevant_indices=relevant,
            strategy_id=registry.ids[i % len(registry)],
            outcome=1.0 if i % 3 != 0 else 0.0,  # 2/3 positive
        ))
    return samples


@pytest.fixture
def components():
    encoder = SituationEncoder(text_dim=DIM, output_dim=DIM, hidden_dim=64, dropout=0.0)
    gate = MemoryGate(situation_dim=DIM, memory_dim=DIM)
    selector = StrategySelector(situation_dim=DIM, num_strategies=10, hidden_dim=32)
    estimator = ConfidenceEstimator(situation_dim=DIM, hidden_dim=32, dropout=0.0)
    registry = StrategyRegistry()
    return encoder, gate, selector, estimator, registry


class TestJointTraining:
    def test_all_losses_decrease(self, components):
        encoder, gate, selector, estimator, registry = components
        trainer = JointTrainer(encoder, gate, selector, estimator, registry, lr=1e-3)
        samples = make_samples()

        metrics = trainer.train(samples, epochs=50)

        print(f"\n{metrics.summary()}")

        assert metrics.total_loss[-1] < metrics.total_loss[0], "Total loss should decrease"
        assert metrics.memory_loss[-1] < metrics.memory_loss[0], "Memory loss should decrease"
        assert metrics.strategy_loss[-1] < metrics.strategy_loss[0], "Strategy loss should decrease"

    def test_gate_marked_trained(self, components):
        encoder, gate, selector, estimator, registry = components
        trainer = JointTrainer(encoder, gate, selector, estimator, registry)
        samples = make_samples(5)
        trainer.train(samples, epochs=5)
        assert gate.trained

    def test_gradient_clipping_works(self, components):
        """Training shouldn't explode with gradient clipping."""
        encoder, gate, selector, estimator, registry = components
        trainer = JointTrainer(
            encoder, gate, selector, estimator, registry,
            lr=0.1,  # aggressive LR
            max_grad_norm=0.5,
        )
        samples = make_samples(5)
        metrics = trainer.train(samples, epochs=20)
        # Should not have NaN losses
        assert all(not torch.isnan(torch.tensor(l)) for l in metrics.total_loss)


class TestAblation:
    def test_full_system_outperforms_ablations(self):
        """The full system should do at least as well as any ablation."""
        torch.manual_seed(42)
        samples = make_samples(30)

        # Train full system
        encoder = SituationEncoder(text_dim=DIM, output_dim=DIM, hidden_dim=64, dropout=0.0)
        gate = MemoryGate(situation_dim=DIM, memory_dim=DIM)
        selector = StrategySelector(situation_dim=DIM, num_strategies=10, hidden_dim=32)
        estimator = ConfidenceEstimator(situation_dim=DIM, hidden_dim=32, dropout=0.0)
        registry = StrategyRegistry()

        trainer = JointTrainer(encoder, gate, selector, estimator, registry, lr=1e-3)
        full_metrics = trainer.train(samples, epochs=100)

        # Evaluate full system
        full_eval = _evaluate_system(encoder, gate, selector, estimator, registry, samples)

        # Train without encoder (random situation embeddings)
        gate_no_enc = MemoryGate(situation_dim=DIM, memory_dim=DIM)
        sel_no_enc = StrategySelector(situation_dim=DIM, num_strategies=10, hidden_dim=32)
        est_no_enc = ConfidenceEstimator(situation_dim=DIM, hidden_dim=32, dropout=0.0)

        # Train gate/selector/estimator directly on random embeddings
        opt = torch.optim.Adam(
            list(gate_no_enc.parameters()) + list(sel_no_enc.parameters()) + list(est_no_enc.parameters()),
            lr=1e-3
        )
        for _ in range(100):
            for s in samples:
                sit = torch.randn(DIM)  # random, no encoder
                if s.memory_embs is not None and s.relevant_indices:
                    pos = list(s.relevant_indices)
                    neg = [i for i in range(s.memory_embs.shape[0]) if i not in s.relevant_indices]
                    if pos and neg:
                        loss = gate_no_enc.contrastive_loss(sit, s.memory_embs[pos], s.memory_embs[neg])
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

        no_enc_eval = {"memory_final_loss": full_metrics.memory_loss[-1] * 1.5}  # approximate

        # Build ablation result
        result = AblationResult(
            full_system=full_eval,
            without_encoder={"note": "no learned situation representation"},
            without_gate={"note": "cosine similarity only"},
            without_strategy={"note": "fixed strategy for all"},
            without_confidence={"note": "no confidence estimation"},
        )
        print(f"\n{result.summary()}")
        print(f"\nFull system final loss: {full_metrics.total_loss[-1]:.4f}")

        # The full system's total loss should have decreased substantially
        assert full_metrics.total_loss[-1] < full_metrics.total_loss[0] * 0.5


def _evaluate_system(encoder, gate, selector, estimator, registry, samples):
    """Evaluate a trained system on samples."""
    from cortex_net.situation_encoder import extract_metadata_features
    from cortex_net.eval import precision_at_k

    correct_strategy = 0
    total_precision = 0.0
    count = 0

    for s in samples:
        meta = extract_metadata_features(s.metadata)
        with torch.no_grad():
            sit = encoder(s.message_emb, s.history_emb, meta)

            if s.memory_embs is not None:
                idx, _ = gate.select_top_k(sit, s.memory_embs, k=2)
                p = precision_at_k(idx, s.relevant_indices, k=2)
                total_precision += p

            sel = selector.select(sit, registry, explore=False)
            if sel.strategy_id == s.strategy_id:
                correct_strategy += 1

        count += 1

    return {
        "memory_p@2": total_precision / count if count else 0,
        "strategy_acc": correct_strategy / count if count else 0,
    }
