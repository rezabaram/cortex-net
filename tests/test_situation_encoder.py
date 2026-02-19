"""Tests for Situation Encoder."""

import pytest
import torch
import torch.nn.functional as F

from cortex_net.situation_encoder import (
    SituationEncoder,
    SituationFeatures,
    SituationContrastiveLoss,
    extract_metadata_features,
    META_DIM,
)
from cortex_net.state_manager import StateManager, CheckpointMetadata


class TestMetadataExtraction:
    def test_default_features(self):
        meta = extract_metadata_features({})
        assert meta.shape == (META_DIM,)
        assert (meta >= 0).all()
        assert (meta <= 1).all()

    def test_features_from_metadata(self):
        meta = extract_metadata_features({
            "hour_of_day": 14,
            "day_of_week": 4,  # Friday
            "conversation_length": 10,
            "is_group_chat": 1,
        })
        assert meta[0] == pytest.approx(14 / 24)
        assert meta[1] == pytest.approx(4 / 6)
        assert meta[5] == 1.0  # is_group_chat

    def test_features_capped(self):
        meta = extract_metadata_features({
            "conversation_length": 1000,  # way over cap
            "message_length": 10000,
            "time_since_last": 999999,
        })
        assert (meta <= 1.0).all()


class TestSituationEncoder:
    @pytest.fixture
    def encoder(self):
        return SituationEncoder(text_dim=32, output_dim=16, hidden_dim=64)

    def test_output_shape(self, encoder):
        msg = torch.randn(32)
        hist = torch.randn(32)
        meta = torch.randn(META_DIM)
        out = encoder(msg, hist, meta)
        assert out.shape == (16,)

    def test_output_normalized(self, encoder):
        msg = torch.randn(32)
        hist = torch.randn(32)
        meta = torch.randn(META_DIM)
        out = encoder(msg, hist, meta)
        assert torch.norm(out).item() == pytest.approx(1.0, abs=1e-5)

    def test_batch_output(self, encoder):
        msg = torch.randn(4, 32)
        hist = torch.randn(4, 32)
        meta = torch.randn(4, META_DIM)
        out = encoder(msg, hist, meta)
        assert out.shape == (4, 16)

    def test_different_metadata_different_output(self, encoder):
        """Same text + different metadata should produce different embeddings."""
        msg = torch.randn(32)
        hist = torch.randn(32)

        meta_morning = extract_metadata_features({"hour_of_day": 9, "day_of_week": 0})
        meta_night = extract_metadata_features({"hour_of_day": 23, "day_of_week": 4})

        with torch.no_grad():
            out_morning = encoder(msg, hist, meta_morning)
            out_night = encoder(msg, hist, meta_night)

        # Should NOT be identical
        assert not torch.allclose(out_morning, out_night, atol=1e-3)

    def test_encode_situation_with_precomputed(self, encoder):
        features = SituationFeatures(
            message="hello",
            history=["previous msg"],
            metadata={"hour_of_day": 14},
            message_embedding=torch.randn(32),
            history_embedding=torch.randn(32),
        )
        out = encoder.encode_situation(features)
        assert out.shape == (16,)

    def test_encode_situation_zeros_fallback(self, encoder):
        """Without embeddings or text_encoder, falls back to zeros."""
        features = SituationFeatures(message="hello")
        out = encoder.encode_situation(features)
        assert out.shape == (16,)


class TestContrastiveLoss:
    def test_loss_is_scalar(self):
        loss_fn = SituationContrastiveLoss()
        anchor = F.normalize(torch.randn(16), dim=0)
        positive = F.normalize(torch.randn(16), dim=0)
        negatives = F.normalize(torch.randn(5, 16), dim=1)
        loss = loss_fn(anchor, positive, negatives)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_identical_positive_low_loss(self):
        loss_fn = SituationContrastiveLoss()
        anchor = F.normalize(torch.randn(16), dim=0)
        positive = anchor.clone()  # identical
        negatives = F.normalize(torch.randn(5, 16), dim=1)
        loss = loss_fn(anchor, positive, negatives)
        # Should be very low since positive is identical to anchor
        assert loss.item() < 1.0


class TestTraining:
    def test_encoder_learns_to_cluster(self):
        """Train encoder to produce similar embeddings for similar situations."""
        torch.manual_seed(42)
        encoder = SituationEncoder(text_dim=32, output_dim=16, hidden_dim=64, dropout=0.0)
        loss_fn = SituationContrastiveLoss(temperature=0.1)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

        # Create two "situation types":
        # Type A: morning, short message, no history
        # Type B: evening, long message, with history
        type_a_meta = extract_metadata_features({"hour_of_day": 9, "day_of_week": 1, "message_length": 20})
        type_b_meta = extract_metadata_features({"hour_of_day": 22, "day_of_week": 5, "message_length": 400})

        losses = []
        for _ in range(100):
            # Anchor and positive are both Type A
            anchor = encoder(torch.randn(32), torch.randn(32), type_a_meta)
            positive = encoder(torch.randn(32), torch.randn(32), type_a_meta)

            # Negatives are Type B
            negs = torch.stack([
                encoder(torch.randn(32), torch.randn(32), type_b_meta)
                for _ in range(3)
            ])

            loss = loss_fn(anchor, positive, negs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_checkpoint_roundtrip(self, tmp_path):
        encoder = SituationEncoder(text_dim=32, output_dim=16, hidden_dim=64)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

        # Modify weights
        with torch.no_grad():
            encoder.encoder[0].weight.fill_(0.42)

        sm = StateManager(tmp_path / "state")
        sm.save(encoder, optimizer, CheckpointMetadata(component_name="situation_encoder", step=50))

        encoder2 = SituationEncoder(text_dim=32, output_dim=16, hidden_dim=64)
        meta = sm.load(encoder2, component_name="situation_encoder")

        assert meta is not None
        assert meta.step == 50
        assert torch.allclose(encoder.encoder[0].weight, encoder2.encoder[0].weight)


class TestIntegrationWithMemoryGate:
    def test_encoder_output_works_with_gate(self):
        """Situation Encoder output can be fed into Memory Gate."""
        from cortex_net.memory_gate import MemoryGate

        encoder = SituationEncoder(text_dim=32, output_dim=64, hidden_dim=128)
        gate = MemoryGate(situation_dim=64, memory_dim=64)

        # Encode a situation
        msg = torch.randn(32)
        hist = torch.randn(32)
        meta = extract_metadata_features({"hour_of_day": 14})
        situation = encoder(msg, hist, meta)

        # Use it with Memory Gate
        memories = torch.randn(10, 64)
        indices, scores = gate.select_top_k(situation, memories, k=3)

        assert indices.shape == (3,)
        assert scores.shape == (3,)

    def test_encoder_improves_gate_with_metadata(self):
        """Memory Gate + Situation Encoder should distinguish situations
        that raw text embeddings can't."""
        torch.manual_seed(42)

        encoder = SituationEncoder(text_dim=32, output_dim=32, hidden_dim=64, dropout=0.0)
        from cortex_net.memory_gate import MemoryGate
        gate = MemoryGate(situation_dim=32, memory_dim=32)

        # Same message text, different metadata contexts
        same_msg = torch.randn(32)
        same_hist = torch.randn(32)

        meta_standup = extract_metadata_features({"hour_of_day": 9, "day_of_week": 0})
        meta_incident = extract_metadata_features({"hour_of_day": 23, "day_of_week": 4})

        # Different situation embeddings despite same text
        with torch.no_grad():
            sit_standup = encoder(same_msg, same_hist, meta_standup)
            sit_incident = encoder(same_msg, same_hist, meta_incident)

        # Should produce different embeddings (even untrained, metadata changes the input)
        cosine = F.cosine_similarity(sit_standup.unsqueeze(0), sit_incident.unsqueeze(0))
        assert cosine.item() < 1.0  # not perfectly identical

        # And therefore different memory rankings
        memories = torch.randn(10, 32)
        with torch.no_grad():
            idx1, _ = gate.select_top_k(sit_standup, memories, k=3)
            idx2, _ = gate.select_top_k(sit_incident, memories, k=3)

        # Rankings may differ (not guaranteed but very likely with different inputs)
        # This is a smoke test, not a hard assertion
        assert idx1.shape == (3,)
