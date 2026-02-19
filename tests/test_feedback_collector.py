"""Tests for feedback collector and replay buffer."""

import pytest
from pathlib import Path

from cortex_net.feedback_collector import (
    extract_feedback,
    FeedbackSignal,
    InteractionOutcome,
    ReplayBuffer,
    ReplayEntry,
)


class TestFeedbackExtraction:
    def test_positive_thanks(self):
        signals = extract_feedback("Thanks, that's exactly what I needed!")
        assert any(s.signal_type == "positive" for s in signals)
        assert all(s.reward > 0 for s in signals if s.signal_type == "positive")

    def test_positive_emoji(self):
        signals = extract_feedback("👍")
        assert any(s.signal_type == "positive" for s in signals)

    def test_negative_wrong(self):
        signals = extract_feedback("That's wrong, the port is 3000 not 8080")
        assert any(s.signal_type == "negative" for s in signals)
        assert all(s.reward < 0 for s in signals if s.signal_type == "negative")

    def test_correction(self):
        signals = extract_feedback("No, I meant the production server")
        types = {s.signal_type for s in signals}
        assert "correction" in types or "negative" in types

    def test_confusion(self):
        signals = extract_feedback("What do you mean? I don't understand")
        assert any(s.signal_type == "confusion" for s in signals)

    def test_neutral_no_signal(self):
        signals = extract_feedback("What's the weather in Amsterdam?")
        # Should have no strong signals
        assert all(abs(s.reward) < 0.5 for s in signals)

    def test_multiple_signals(self):
        signals = extract_feedback("That's wrong! What do you mean?")
        types = {s.signal_type for s in signals}
        assert len(types) >= 2  # both negative and confusion


class TestInteractionOutcome:
    def test_positive_outcome(self):
        signals = extract_feedback("Perfect, thanks!")
        outcome = InteractionOutcome(signals=signals)
        assert outcome.reward > 0
        assert outcome.confidence_target > 0.5
        assert outcome.memory_was_relevant
        assert outcome.strategy_was_correct

    def test_negative_outcome(self):
        signals = extract_feedback("That's completely wrong")
        outcome = InteractionOutcome(signals=signals)
        assert outcome.reward < 0
        assert outcome.confidence_target < 0.5
        assert not outcome.memory_was_relevant

    def test_no_signal_continued(self):
        outcome = InteractionOutcome(
            signals=[], conversation_continued=True, topic_switched=False
        )
        assert outcome.reward == 0.1  # mild positive

    def test_no_signal_left(self):
        outcome = InteractionOutcome(
            signals=[], conversation_continued=False, topic_switched=False
        )
        assert outcome.reward == 0.0


class TestReplayBuffer:
    def test_add_and_len(self, tmp_path):
        buf = ReplayBuffer(capacity=100, path=tmp_path / "buf.jsonl")
        entry = ReplayEntry(
            situation_emb=[0.1, 0.2, 0.3],
            memory_indices=[0, 1],
            memory_scores=[0.9, 0.7],
            strategy_id="quick_answer",
            confidence=0.85,
            reward=0.7,
            confidence_target=0.85,
            memory_was_relevant=True,
            strategy_was_correct=True,
        )
        buf.add(entry)
        assert len(buf) == 1

    def test_capacity_limit(self, tmp_path):
        buf = ReplayBuffer(capacity=5, path=tmp_path / "buf.jsonl")
        for i in range(10):
            buf.add(ReplayEntry(
                situation_emb=[float(i)],
                memory_indices=[],
                memory_scores=[],
                strategy_id="",
                confidence=0.5,
                reward=0.0,
                confidence_target=0.5,
                memory_was_relevant=False,
                strategy_was_correct=True,
            ))
        assert len(buf) == 5

    def test_persistence(self, tmp_path):
        path = tmp_path / "buf.jsonl"
        buf1 = ReplayBuffer(capacity=100, path=path)
        buf1.add(ReplayEntry(
            situation_emb=[1.0, 2.0],
            memory_indices=[0],
            memory_scores=[0.9],
            strategy_id="deep_research",
            confidence=0.9,
            reward=0.8,
            confidence_target=0.9,
            memory_was_relevant=True,
            strategy_was_correct=True,
        ))

        # Load in new buffer
        buf2 = ReplayBuffer(capacity=100, path=path)
        assert len(buf2) == 1
        assert buf2.entries[0].strategy_id == "deep_research"

    def test_sample(self, tmp_path):
        buf = ReplayBuffer(capacity=100, path=tmp_path / "buf.jsonl")
        for i in range(20):
            buf.add(ReplayEntry(
                situation_emb=[float(i)],
                memory_indices=[],
                memory_scores=[],
                strategy_id="",
                confidence=0.5,
                reward=0.0,
                confidence_target=0.5,
                memory_was_relevant=False,
                strategy_was_correct=True,
            ))
        sampled = buf.sample(5)
        assert len(sampled) == 5
