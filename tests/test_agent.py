"""Tests for the live agent (mocked Claude API)."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from cortex_net.agent import CortexAgent, AgentConfig


@pytest.fixture
def mock_openai():
    """Mock the OpenAI client."""
    with patch("cortex_net.agent.OpenAI") as mock:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Here's your answer about deployment."))]
        mock.return_value.chat.completions.create.return_value = mock_response
        yield mock


@pytest.fixture
def agent(tmp_path, mock_openai):
    config = AgentConfig(
        state_dir=str(tmp_path / "state"),
        model="claude-sonnet-4-20250514",
        online_learning=True,
        update_every=2,
    )
    return CortexAgent(config=config, api_key="test-key")


class TestAgent:
    def test_chat_returns_response(self, agent):
        response = agent.chat("How should I deploy?")
        assert response == "Here's your answer about deployment."

    def test_chat_builds_history(self, agent):
        agent.chat("Hello")
        agent.chat("How are you?")
        assert len(agent.history) == 4  # 2 user + 2 assistant

    def test_memories_seeded(self, agent):
        agent.add_memories([
            "We use Kubernetes for deployments",
            "Last deploy caused an outage",
        ])
        assert len(agent.memory_bank) == 2

    def test_memories_used_in_prompt(self, agent, mock_openai):
        agent.add_memories(["We use Kubernetes for deployments"])
        agent.chat("How should I deploy?")

        call_args = mock_openai.return_value.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", [])
        system = messages[0]["content"] if messages else ""
        assert "Kubernetes" in system

    def test_strategy_in_prompt(self, agent, mock_openai):
        agent.chat("Help me brainstorm names for our tool")

        call_args = mock_openai.return_value.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages", [])
        system = messages[0]["content"] if messages else ""
        assert "Approach:" in system

    def test_feedback_loop(self, agent):
        agent.chat("What port does Redis use?")
        # Second message extracts feedback from "Thanks"
        agent.chat("Thanks, that's perfect!")

        if agent.online_trainer:
            assert agent.online_trainer.buffer_size >= 1

    def test_save_and_load(self, tmp_path, mock_openai):
        config = AgentConfig(
            state_dir=str(tmp_path / "state"),
            online_learning=False,
        )
        agent1 = CortexAgent(config=config, api_key="test-key")
        agent1.add_memories(["Test memory"])
        agent1.chat("Hello")
        agent1.save()

        # Load in new agent
        agent2 = CortexAgent(config=config, api_key="test-key")
        assert len(agent2.memory_bank) >= 1  # at least the seeded memory persists
        assert len(agent2.history) == 2

    def test_stats(self, agent):
        agent.chat("Hello")
        stats = agent.stats()
        assert stats["conversation_turns"] == 2
        assert "memories" in stats
        assert "gate_trained" in stats

    def test_auto_memorize(self, agent, mock_openai):
        # Mock a long response so assistant side gets memorized
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Here's a detailed explanation " * 10))]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        initial = len(agent.memory_bank)
        # Long message triggers user-side memorization, long response triggers assistant-side
        agent.chat("Can you explain the entire deployment process including CI/CD, staging, and production rollout?")
        assert len(agent.memory_bank) > initial


class TestMemoryBankIntegration:
    def test_agent_uses_memory_bank(self, agent, mock_openai):
        """Agent should use MemoryBank (SQLite) not old MemoryStore."""
        agent.add_memories(["K8s deployment info"])
        assert hasattr(agent, 'memory_bank')
        assert len(agent.memory_bank) == 1
        stats = agent.memory_bank.stats()
        assert stats["sources"]["seeded"] == 1
