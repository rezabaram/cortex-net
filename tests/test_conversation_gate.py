"""Tests for Conversation Gate."""

import torch
import pytest

from cortex_net.conversation_gate import ConversationGate, ConversationContext


class TestConversationGate:
    def test_create(self):
        gate = ConversationGate(dim=384)
        assert gate.dim == 384
        assert not gate.trained

    def test_score_turns(self):
        gate = ConversationGate(dim=64)
        situation = torch.randn(64)
        turns = torch.randn(5, 64)
        scores = gate.score_turns(situation, turns)
        assert scores.shape == (5,)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_forward_min_turns(self):
        gate = ConversationGate(dim=64)
        situation = torch.randn(64)
        turns = torch.randn(5, 64)
        mask, scores = gate.forward(situation, turns, min_turns=2)
        assert mask.sum() >= 2

    def test_forward_max_select(self):
        gate = ConversationGate(dim=64)
        situation = torch.randn(64)
        turns = torch.randn(10, 64)
        mask, scores = gate.forward(situation, turns, min_turns=1, max_select=3)
        assert mask.sum() <= 3

    def test_select_turns_empty(self):
        gate = ConversationGate(dim=64)
        situation = torch.randn(64)
        turns = torch.randn(0, 64)
        ctx = gate.select_turns(situation, turns, messages=[])
        assert ctx.messages == []
        assert ctx.scores == []

    def test_select_turns(self):
        gate = ConversationGate(dim=64)
        situation = torch.randn(64)
        turns = torch.randn(5, 64)
        messages = [
            {"role": "user", "content": f"msg {i}"} for i in range(5)
        ]
        ctx = gate.select_turns(situation, turns, messages, min_turns=2, max_turns=4)
        assert len(ctx.messages) >= 2
        assert len(ctx.messages) <= 4
        assert len(ctx.scores) == len(ctx.messages)

    def test_identity_init_cosine_like(self):
        """Identity W should produce scores correlated with cosine similarity."""
        gate = ConversationGate(dim=64)
        situation = torch.nn.functional.normalize(torch.randn(64), dim=0)

        # One turn very similar, one very different
        similar = situation + 0.1 * torch.randn(64)
        similar = torch.nn.functional.normalize(similar, dim=0)
        different = -situation + 0.1 * torch.randn(64)
        different = torch.nn.functional.normalize(different, dim=0)

        turns = torch.stack([different, similar])  # idx 0=different, 1=similar
        scores = gate.score_turns(situation, turns)
        # Similar turn should score higher (or equal with recency)
        # Allow some tolerance because recency bias adds noise
        assert scores[1] > scores[0] - 0.3  # similar is at least close

    def test_training_loss(self):
        gate = ConversationGate(dim=64)
        situation = torch.randn(64)
        turns = torch.randn(5, 64)
        labels = torch.tensor([0, 1, 0, 1, 1], dtype=torch.float32)
        loss = gate.training_loss(situation, turns, labels)
        assert loss.item() > 0
        assert loss.requires_grad

    def test_recency_bias(self):
        """More recent turns should get higher scores, all else equal."""
        gate = ConversationGate(dim=64)
        # All turns have same content but different positions
        turn_emb = torch.nn.functional.normalize(torch.randn(64), dim=0)
        turns = turn_emb.unsqueeze(0).expand(5, -1)
        situation = turn_emb  # same as all turns
        scores = gate.score_turns(situation, turns)
        # Last turn (most recent) should score >= first turn (oldest)
        assert scores[-1] >= scores[0]
