"""Tests for Confidence Estimator."""

import pytest
import torch

from cortex_net.confidence_estimator import (
    ConfidenceEstimator,
    ConfidenceResult,
    ContextSummary,
    CalibrationLoss,
    expected_calibration_error,
    CONTEXT_SUMMARY_DIM,
)
from cortex_net.state_manager import StateManager, CheckpointMetadata


class TestContextSummary:
    def test_to_tensor_shape(self):
        ctx = ContextSummary(num_memories_retrieved=5, top_memory_score=0.9)
        t = ctx.to_tensor()
        assert t.shape == (CONTEXT_SUMMARY_DIM,)

    def test_values_normalized(self):
        ctx = ContextSummary(
            num_memories_retrieved=100,  # over cap
            query_length=10000,
            history_length=999,
        )
        t = ctx.to_tensor()
        assert (t <= 1.0).all()
        assert (t >= 0.0).all()


class TestConfidenceEstimator:
    @pytest.fixture
    def estimator(self):
        return ConfidenceEstimator(situation_dim=32, hidden_dim=64)

    def test_output_shape(self, estimator):
        sit = torch.randn(32)
        ctx = torch.randn(CONTEXT_SUMMARY_DIM)
        conf = estimator(sit, ctx)
        assert conf.shape == ()
        assert 0 <= conf.item() <= 1

    def test_batch_output(self, estimator):
        sit = torch.randn(4, 32)
        ctx = torch.randn(4, CONTEXT_SUMMARY_DIM)
        conf = estimator(sit, ctx)
        assert conf.shape == (4,)
        assert (conf >= 0).all() and (conf <= 1).all()

    def test_estimate_returns_result(self, estimator):
        sit = torch.randn(32)
        ctx = ContextSummary(num_memories_retrieved=5, top_memory_score=0.9)
        result = estimator.estimate(sit, ctx)
        assert isinstance(result, ConfidenceResult)
        assert 0 <= result.confidence <= 1
        assert result.action in ("proceed", "hedge", "escalate")

    def test_action_thresholds(self):
        assert ConfidenceResult.action_from_confidence(0.9) == "proceed"
        assert ConfidenceResult.action_from_confidence(0.6) == "hedge"
        assert ConfidenceResult.action_from_confidence(0.2) == "escalate"


class TestCalibrationLoss:
    def test_perfect_calibration_low_loss(self):
        loss_fn = CalibrationLoss()
        # Predicted matches actual perfectly
        predicted = torch.tensor([0.9, 0.1, 0.8, 0.2])
        actual = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = loss_fn(predicted, actual)
        assert loss.item() < 0.5  # should be low

    def test_overconfident_high_loss(self):
        loss_fn = CalibrationLoss()
        # Always confident but often wrong
        predicted = torch.tensor([0.95, 0.95, 0.95, 0.95])
        actual = torch.tensor([1.0, 0.0, 0.0, 0.0])
        loss = loss_fn(predicted, actual)
        assert loss.item() > 0.5  # should be high


class TestECE:
    def test_perfect_calibration(self):
        predicted = torch.tensor([0.1, 0.3, 0.7, 0.9])
        actual = torch.tensor([0.0, 0.0, 1.0, 1.0])
        ece = expected_calibration_error(predicted, actual, num_bins=5)
        assert ece < 0.3  # reasonably calibrated

    def test_terrible_calibration(self):
        # Always predicts 0.9 but only right 25% of the time
        predicted = torch.tensor([0.9] * 100)
        actual = torch.cat([torch.ones(25), torch.zeros(75)])
        ece = expected_calibration_error(predicted, actual, num_bins=10)
        assert ece > 0.5


class TestTraining:
    def test_learns_calibration(self):
        """Estimator should learn to output high confidence for easy cases,
        low confidence for hard cases."""
        torch.manual_seed(42)
        estimator = ConfidenceEstimator(situation_dim=16, hidden_dim=32, dropout=0.0)
        loss_fn = CalibrationLoss()
        optimizer = torch.optim.Adam(estimator.parameters(), lr=1e-3)

        # Easy situations: good context → correct (1.0)
        # Hard situations: bad context → incorrect (0.0)
        easy_sits = torch.randn(20, 16) + 1.0  # shifted
        hard_sits = torch.randn(20, 16) - 1.0

        easy_ctx = torch.tensor([[0.5, 0.9, 0.8, 0.3, 0.8, 0.2, 0.1]] * 20)
        hard_ctx = torch.tensor([[0.5, 0.2, 0.1, 0.05, 0.3, 0.8, 0.0]] * 20)

        easy_outcomes = torch.ones(20)
        hard_outcomes = torch.zeros(20)

        sits = torch.cat([easy_sits, hard_sits])
        ctxs = torch.cat([easy_ctx, hard_ctx])
        outcomes = torch.cat([easy_outcomes, hard_outcomes])

        losses = []
        for epoch in range(200):
            predicted = estimator(sits, ctxs)
            loss = loss_fn(predicted, outcomes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0]

        # Check: easy should be high confidence, hard should be low
        with torch.no_grad():
            easy_conf = estimator(easy_sits, easy_ctx).mean().item()
            hard_conf = estimator(hard_sits, hard_ctx).mean().item()

        assert easy_conf > hard_conf, f"Easy ({easy_conf:.2f}) should be more confident than hard ({hard_conf:.2f})"

        # Check ECE
        with torch.no_grad():
            all_preds = estimator(sits, ctxs)
        ece = expected_calibration_error(all_preds, outcomes)
        print(f"\nECE: {ece:.4f}  Easy conf: {easy_conf:.2f}  Hard conf: {hard_conf:.2f}")
        assert ece < 0.2  # reasonably calibrated


class TestCheckpointing:
    def test_roundtrip(self, tmp_path):
        est = ConfidenceEstimator(situation_dim=16, hidden_dim=32)
        sm = StateManager(tmp_path / "state")
        sm.save(est, None, CheckpointMetadata(component_name="confidence_estimator", step=5))

        est2 = ConfidenceEstimator(situation_dim=16, hidden_dim=32)
        meta = sm.load(est2, component_name="confidence_estimator")
        assert meta is not None
        for p1, p2 in zip(est.parameters(), est2.parameters()):
            assert torch.allclose(p1, p2)


class TestIntegration:
    def test_full_pipeline(self):
        """Situation Encoder → Memory Gate → Confidence Estimator."""
        from cortex_net.situation_encoder import SituationEncoder, extract_metadata_features
        from cortex_net.memory_gate import MemoryGate

        encoder = SituationEncoder(text_dim=32, output_dim=64, hidden_dim=128)
        gate = MemoryGate(situation_dim=64, memory_dim=64)
        estimator = ConfidenceEstimator(situation_dim=64, hidden_dim=64)

        msg = torch.randn(32)
        hist = torch.randn(32)
        meta = extract_metadata_features({"hour_of_day": 14})
        memories = torch.randn(10, 64)

        with torch.no_grad():
            sit = encoder(msg, hist, meta)
            indices, scores = gate.select_top_k(sit, memories, k=3)

        ctx = ContextSummary(
            num_memories_retrieved=3,
            top_memory_score=scores[0].item(),
            mean_memory_score=scores.mean().item(),
            score_spread=(scores[0] - scores[-1]).item(),
        )
        result = estimator.estimate(sit, ctx)

        assert 0 <= result.confidence <= 1
        assert result.action in ("proceed", "hedge", "escalate")
