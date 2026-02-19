"""Tests for the agent monitoring system."""

import json
import pytest
from pathlib import Path

from cortex_net.monitor import AgentMonitor, InteractionLog


class TestInteractionLog:
    def test_default_values(self):
        log = InteractionLog()
        assert log.turn_number == 0
        assert log.confidence == 0.0
        assert log.tool_calls == 0
        assert log.feedback_reward is None

    def test_set_values(self):
        log = InteractionLog(
            turn_number=5,
            strategy_id="debug",
            confidence=0.85,
            memories_retrieved=3,
            tool_calls=2,
            feedback_reward=0.7,
            feedback_type="positive",
        )
        assert log.strategy_id == "debug"
        assert log.confidence == 0.85
        assert log.feedback_reward == 0.7


class TestAgentMonitor:
    def test_create(self, tmp_path):
        monitor = AgentMonitor(log_dir=tmp_path, name="test")
        assert monitor.jsonl_path == tmp_path / "test_interactions.jsonl"

    def test_new_interaction_increments(self, tmp_path):
        monitor = AgentMonitor(log_dir=tmp_path, name="test")
        i1 = monitor.new_interaction()
        i2 = monitor.new_interaction()
        assert i1.turn_number == 1
        assert i2.turn_number == 2
        assert i1.timestamp > 0

    def test_record_writes_jsonl(self, tmp_path):
        monitor = AgentMonitor(log_dir=tmp_path, name="test")
        ilog = monitor.new_interaction()
        ilog.strategy_id = "implement"
        ilog.confidence = 0.75
        ilog.response_length = 500
        ilog.total_ms = 1234.5
        monitor.record(ilog)

        # Verify file
        lines = monitor.jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["strategy_id"] == "implement"
        assert data["confidence"] == 0.75
        assert data["response_length"] == 500

    def test_record_multiple(self, tmp_path):
        monitor = AgentMonitor(log_dir=tmp_path, name="test")
        for i in range(5):
            ilog = monitor.new_interaction()
            ilog.confidence = 0.5 + i * 0.1
            monitor.record(ilog)

        lines = monitor.jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_feedback_tracking(self, tmp_path):
        monitor = AgentMonitor(log_dir=tmp_path, name="test")

        # Positive
        ilog = monitor.new_interaction()
        ilog.feedback_type = "positive"
        ilog.feedback_reward = 0.7
        monitor.record(ilog)

        # Negative
        ilog = monitor.new_interaction()
        ilog.feedback_type = "negative"
        ilog.feedback_reward = -0.5
        monitor.record(ilog)

        # Correction
        ilog = monitor.new_interaction()
        ilog.feedback_type = "correction"
        ilog.feedback_reward = 0.3
        monitor.record(ilog)

        s = monitor.summary()
        assert s["positive_feedback"] == 1
        assert s["negative_feedback"] == 1
        assert s["corrections"] == 1
        assert s["total_interactions"] == 3

    def test_summary_text(self, tmp_path):
        monitor = AgentMonitor(log_dir=tmp_path, name="atlas")
        ilog = monitor.new_interaction()
        ilog.confidence = 0.8
        ilog.tool_calls = 3
        ilog.total_ms = 5000
        monitor.record(ilog)

        text = monitor.summary_text()
        assert "atlas" in text
        assert "1 interactions" in text
        assert "3 tool calls" in text

    def test_resume_count(self, tmp_path):
        # Write some existing data
        path = tmp_path / "test_interactions.jsonl"
        for i in range(3):
            with open(path, "a") as f:
                f.write(json.dumps({"turn": i}) + "\n")

        monitor = AgentMonitor(log_dir=tmp_path, name="test")
        assert monitor._total_interactions == 3
        ilog = monitor.new_interaction()
        assert ilog.turn_number == 4  # continues from existing
