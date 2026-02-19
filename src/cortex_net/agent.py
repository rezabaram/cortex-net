"""Live Agent — cortex-net wrapped around Claude.

A minimal but complete agent loop:
1. Receive user message
2. Assemble context via cortex-net (situation, memories, strategy, confidence)
3. Send assembled prompt to Claude
4. Return response
5. On next message, extract feedback from user reaction and update components

Usage:
    agent = CortexAgent(state_dir="./state")
    response = agent.chat("How should I deploy the new service?")
    # ... user reads response, sends next message ...
    response = agent.chat("Thanks, that worked!")  # ← feedback extracted from this
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
import torch
from sentence_transformers import SentenceTransformer

from cortex_net.confidence_estimator import ConfidenceEstimator, ContextSummary
from cortex_net.feedback_collector import ReplayBuffer, extract_feedback, InteractionOutcome
from cortex_net.memory_bank import MemoryBank
from cortex_net.memory_gate import MemoryGate
from cortex_net.online_trainer import OnlineTrainer
from cortex_net.situation_encoder import SituationEncoder, extract_metadata_features
from cortex_net.state_manager import StateManager
from cortex_net.strategy_selector import StrategySelector, StrategyRegistry
from cortex_net.tools import ToolRegistry, create_default_tools
from cortex_net.monitor import AgentMonitor, InteractionLog


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float = 0.0


@dataclass
class AgentConfig:
    """Configuration for the live agent."""

    # Model (OpenAI-compatible API)
    model: str = "MiniMax-M1"
    base_url: str = "https://api.minimax.io/v1"
    max_tokens: int = 1024

    # cortex-net dimensions
    text_dim: int = 384
    embedding_model: str = "all-MiniLM-L6-v2"

    # Memory
    max_memories: int = 1000
    retrieval_k: int = 5

    # Online learning
    online_learning: bool = True
    update_every: int = 5
    replay_capacity: int = 10000

    # State
    state_dir: str = "./state"

    # Strategy
    strategy_set: str | None = None  # "developer", "support", "generic", or None for default

    # Tools
    tools_enabled: bool = False
    tools_workdir: str = "."
    max_tool_rounds: int = 25  # max tool call rounds per chat()
    chat_timeout: int = 120  # max seconds per chat() call

    # System prompt base
    system_prompt: str = (
        "You are a helpful assistant. Use the provided context to give "
        "accurate, relevant responses."
    )


class MemoryStore:
    """Simple persistent memory store for the agent.

    Stores text memories with their embeddings. In production, this would
    be backed by a proper vector database.
    """

    def __init__(self, path: Path, text_encoder: SentenceTransformer, device: str = "cpu"):
        self.path = path
        self.text_encoder = text_encoder
        self.device = device
        self.texts: list[str] = []
        self.embeddings: torch.Tensor | None = None
        self._load()

    def add(self, text: str) -> None:
        """Add a memory."""
        self.texts.append(text)
        emb = self.text_encoder.encode(text, convert_to_tensor=True).to(self.device)
        if self.embeddings is None:
            self.embeddings = emb.unsqueeze(0)
        else:
            self.embeddings = torch.cat([self.embeddings, emb.unsqueeze(0)], dim=0)
        self._save()

    def add_many(self, texts: list[str]) -> None:
        """Add multiple memories."""
        for t in texts:
            self.add(t)

    def get_all_embeddings(self) -> torch.Tensor | None:
        """Get all memory embeddings."""
        return self.embeddings

    def get_texts(self, indices: list[int]) -> list[str]:
        """Get memory texts by indices."""
        return [self.texts[i] for i in indices if i < len(self.texts)]

    def __len__(self) -> int:
        return len(self.texts)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"texts": self.texts}
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f)
        tmp.rename(self.path)

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path) as f:
            data = json.load(f)
        self.texts = data.get("texts", [])
        if self.texts:
            self.embeddings = self.text_encoder.encode(
                self.texts, convert_to_tensor=True
            ).to(self.device)


class CortexAgent:
    """A live agent powered by cortex-net and Claude.

    The agent maintains conversation history, a persistent memory store,
    and all four cortex-net components. It learns from every interaction.

    Args:
        config: Agent configuration.
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.device = "cpu"

        # LLM client (OpenAI-compatible)
        self.client = OpenAI(
            api_key=api_key or os.environ.get("LLM_API_KEY", ""),
            base_url=self.config.base_url,
        )

        # Text encoder
        self.text_encoder = SentenceTransformer(
            self.config.embedding_model, device=self.device
        )

        # cortex-net components
        dim = self.config.text_dim
        self.encoder = SituationEncoder(
            text_dim=dim, output_dim=dim, hidden_dim=512, dropout=0.1
        ).to(self.device)
        self.gate = MemoryGate(
            situation_dim=dim, memory_dim=dim
        ).to(self.device)
        # Strategy registry (configurable per agent type)
        strategy_set = getattr(self.config, 'strategy_set', None)
        if strategy_set:
            from cortex_net.strategy_selector import get_strategy_set
            self.registry = StrategyRegistry(get_strategy_set(strategy_set))
        else:
            self.registry = StrategyRegistry()

        self.selector = StrategySelector(
            situation_dim=dim, num_strategies=len(self.registry), hidden_dim=128
        ).to(self.device)
        self.estimator = ConfidenceEstimator(
            situation_dim=dim, hidden_dim=64, dropout=0.1
        ).to(self.device)

        # State management
        state_dir = Path(self.config.state_dir)
        self.state_mgr = StateManager(state_dir)

        # Memory bank (SQLite-backed, extensible)
        self.memory_bank = MemoryBank(
            db_path=state_dir / "memory.db",
            blob_dir=state_dir / "blobs",
        )

        # Online trainer
        replay_buf = ReplayBuffer(
            capacity=self.config.replay_capacity,
            path=state_dir / "replay_buffer.jsonl",
        )
        self.online_trainer = OnlineTrainer(
            self.encoder, self.gate, self.selector, self.estimator, self.registry,
            buffer=replay_buf,
            update_every=self.config.update_every,
        ) if self.config.online_learning else None

        # Tools
        self.tool_registry: ToolRegistry | None = None
        if self.config.tools_enabled:
            self.tool_registry = create_default_tools(workdir=self.config.tools_workdir)

        # Monitor
        self.monitor = AgentMonitor(log_dir=state_dir, name=self.config.model.replace("/", "-"))

        # Conversation state
        self.history: list[ConversationTurn] = []
        self._last_situation_emb: torch.Tensor | None = None
        self._last_memory_indices: list[int] = []
        self._last_memory_scores: list[float] = []
        self._last_strategy_id: str = ""
        self._last_confidence: float = 0.5

        # Load saved state
        self._load_state()

    def chat(self, message: str, metadata: dict | None = None) -> str:
        """Send a message and get a response.

        This is the main entry point. Each call:
        1. Extracts feedback from the message (if there was a previous turn)
        2. Assembles context via cortex-net
        3. Calls Claude with the assembled context
        4. Stores the interaction for future learning

        Args:
            message: User message.
            metadata: Optional metadata (hour_of_day, etc.). Auto-populated if not provided.

        Returns:
            Assistant response text.
        """
        ilog = self.monitor.new_interaction()
        ilog.user_message = message[:200]
        ilog.message_length = len(message)
        t_start = time.time()

        # Auto-populate metadata
        if metadata is None:
            now = time.localtime()
            metadata = {
                "hour_of_day": now.tm_hour,
                "day_of_week": now.tm_wday,
            }
        metadata["conversation_length"] = len(self.history) // 2
        metadata["message_length"] = len(message)

        # Step 1: Extract feedback from this message about the PREVIOUS turn
        if self.online_trainer and self._last_situation_emb is not None:
            outcome = self.online_trainer.record_interaction(
                situation_emb=self._last_situation_emb,
                memory_indices=self._last_memory_indices,
                memory_scores=self._last_memory_scores,
                strategy_id=self._last_strategy_id,
                confidence=self._last_confidence,
                user_response=message,
                query=self.history[-2].content if len(self.history) >= 2 else "",
                conversation_continued=True,
            )
            ilog.feedback_reward = outcome.reward
            ilog.feedback_type = outcome.signals[0].signal_type if outcome.signals else "neutral"
            ilog.buffer_size = len(self.online_trainer.buffer)
            ilog.ema_loss = self.online_trainer.ema_loss
            ilog.online_update = self.online_trainer._interactions_since_update == 0

        # Step 2: Encode situation
        t_encode = time.time()
        msg_emb = self.text_encoder.encode(message, convert_to_tensor=True).to(self.device)

        # History embedding (mean of recent messages)
        recent = [t.content for t in self.history[-6:]]  # last 3 turns
        if recent:
            hist_emb = self.text_encoder.encode(
                recent, convert_to_tensor=True
            ).to(self.device).mean(dim=0)
        else:
            hist_emb = torch.zeros(self.config.text_dim, device=self.device)

        meta_tensor = extract_metadata_features(metadata, message, recent).to(self.device)

        with torch.no_grad():
            situation = self.encoder(msg_emb, hist_emb, meta_tensor)
        ilog.situation_norm = situation.norm().item()
        ilog.encoding_ms = (time.time() - t_encode) * 1000

        # Step 3: Memory retrieval via MemoryBank
        t_retrieve = time.time()
        selected_memories = []
        memory_indices = []
        memory_scores = []

        if len(self.memory_bank) > 0:
            results = self.memory_bank.retrieve(
                situation, k=self.config.retrieval_k, gate=self.gate
            )
            selected_memories = [r.memory.text for r in results]
            memory_indices = list(range(len(results)))
            memory_scores = [r.score for r in results]

        ilog.retrieval_ms = (time.time() - t_retrieve) * 1000
        ilog.memories_retrieved = len(selected_memories)
        ilog.memory_scores = [round(s, 3) for s in memory_scores]
        ilog.memory_texts = [m[:80] for m in selected_memories]
        ilog.top_memory_score = memory_scores[0] if memory_scores else 0
        ilog.mean_memory_score = sum(memory_scores) / len(memory_scores) if memory_scores else 0

        # Step 4: Strategy selection
        t_strategy = time.time()
        with torch.no_grad():
            selection = self.selector.select(situation, self.registry, explore=True)
        strategy = selection.strategy
        ilog.strategy_ms = (time.time() - t_strategy) * 1000
        ilog.strategy_id = selection.strategy_id
        ilog.strategy_confidence = selection.confidence
        ilog.strategy_probabilities = {k: round(v, 3) for k, v in selection.probabilities.items()}

        # Step 5: Confidence estimation
        ctx_summary = ContextSummary(
            top_memory_score=memory_scores[0] if memory_scores else 0,
            mean_memory_score=sum(memory_scores) / len(memory_scores) if memory_scores else 0,
            num_memories_retrieved=len(selected_memories),
        )
        with torch.no_grad():
            confidence = self.estimator(
                situation, ctx_summary.to_tensor().to(self.device)
            ).item()

        ilog.confidence = confidence
        ilog.confidence_action = "proceed" if confidence >= 0.8 else "hedge" if confidence >= 0.4 else "escalate"

        # Step 6: Build prompt
        system_parts = [self.config.system_prompt]

        if selected_memories:
            system_parts.append("\n## Relevant Context")
            for i, mem in enumerate(selected_memories, 1):
                system_parts.append(f"{i}. {mem}")

        if strategy:
            system_parts.append(f"\n## Approach: {strategy.name}")
            system_parts.append(strategy.description)

        if confidence < 0.4:
            system_parts.append(
                "\n## Note: Low confidence in available context. "
                "Be upfront about uncertainty and ask for clarification if needed."
            )
        elif confidence < 0.8:
            system_parts.append(
                "\n## Note: Moderate confidence. Include caveats where appropriate."
            )

        system_prompt = "\n".join(system_parts)

        # Step 7: Build messages (conversation history + current)
        messages = []
        for turn in self.history[-4:]:  # last 2 exchanges (test: does shorter context improve task-following?)
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": message})

        # Step 8: Call LLM with tool loop
        t_llm = time.time()
        tool_call_count = 0
        tool_names_used = []
        all_messages = [{"role": "system", "content": system_prompt}] + messages

        call_kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": all_messages,
        }
        if self.tool_registry:
            call_kwargs["tools"] = self.tool_registry.to_openai_tools()

        assistant_text = ""
        t_tool_start = time.time()
        for _round in range(self.config.max_tool_rounds):
            # Timeout check
            if time.time() - t_tool_start > self.config.chat_timeout:
                import logging
                logging.getLogger("cortex-agent").warning(
                    f"Chat timeout after {_round} tool rounds ({self.config.chat_timeout}s)"
                )
                assistant_text = assistant_text or "(timed out — too many tool calls)"
                break
            response = self.client.chat.completions.create(**call_kwargs)
            choice = response.choices[0]

            # If no tool calls, we're done
            if not choice.message.tool_calls:
                assistant_text = choice.message.content or ""
                break

            # Process tool calls
            all_messages.append(choice.message)
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_call_count += 1
                tool_names_used.append(tc.function.name)
                result = self.tool_registry.execute(tc.function.name, args)
                all_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result[:4000],  # cap tool output
                })

            # Update kwargs for next round
            call_kwargs["messages"] = all_messages
        else:
            # Hit max rounds — get whatever the last response was
            assistant_text = choice.message.content or "(max tool rounds reached)"

        ilog.llm_ms = (time.time() - t_llm) * 1000
        ilog.response_length = len(assistant_text)
        ilog.tool_calls = tool_call_count
        ilog.tool_names = tool_names_used
        ilog.total_ms = (time.time() - t_start) * 1000
        self.monitor.record(ilog)

        # Step 9: Update conversation history
        self.history.append(ConversationTurn(role="user", content=message, timestamp=time.time()))
        self.history.append(ConversationTurn(role="assistant", content=assistant_text, timestamp=time.time()))

        # Step 10: Store state for feedback on next turn
        self._last_situation_emb = situation
        self._last_memory_indices = memory_indices
        self._last_memory_scores = memory_scores
        self._last_strategy_id = selection.strategy_id
        self._last_confidence = confidence

        # Step 11: Auto-memorize important exchanges (simple heuristic)
        if len(message) > 50:  # non-trivial messages become memories
            emb = self.text_encoder.encode(message[:200], convert_to_tensor=True)
            self.memory_bank.add_text(
                f"User asked: {message[:200]}", emb, source="user", importance=0.5
            )
        if len(assistant_text) > 100:
            summary = f"I responded about: {message[:100]} → {assistant_text[:200]}"
            emb = self.text_encoder.encode(summary, convert_to_tensor=True)
            self.memory_bank.add_text(summary, emb, source="assistant", importance=0.4)

        # Save state periodically
        if len(self.history) % 10 == 0:
            self._save_state()

        return assistant_text

    def add_memories(self, memories: list[str]) -> None:
        """Seed the agent with initial memories."""
        for text in memories:
            emb = self.text_encoder.encode(text, convert_to_tensor=True)
            self.memory_bank.add_text(text, emb, source="seeded", importance=0.7)

    def save(self) -> None:
        """Save all state to disk."""
        self._save_state()

    def stats(self) -> dict:
        """Get agent statistics."""
        return {
            "conversation_turns": len(self.history),
            "memories": len(self.memory_bank),
            "replay_buffer": len(self.online_trainer.buffer) if self.online_trainer else 0,
            "online_updates": self.online_trainer.total_updates if self.online_trainer else 0,
            "ema_loss": self.online_trainer.ema_loss if self.online_trainer else 0,
            "gate_trained": self.gate.trained,
        }

    def health(self) -> dict:
        """Health check — quick overview of agent state."""
        return {
            "model": self.config.model,
            "strategy_set": self.config.strategy_set or "generic",
            "memories": len(self.memory_bank),
            "replay_buffer": len(self.online_trainer.buffer) if self.online_trainer else 0,
            "online_updates": self.online_trainer.total_updates if self.online_trainer else 0,
            "ema_loss": round(self.online_trainer.ema_loss, 4) if self.online_trainer else 0,
            "gate_trained": self.gate.trained,
            "confidence_avg": round(self.monitor._avg_confidence, 3),
            "total_interactions": self.monitor._total_interactions,
            "tools_enabled": self.tool_registry is not None,
        }

    def _save_state(self) -> None:
        """Save all component weights."""
        state_dir = Path(self.config.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)

        # Save component weights directly (bypass StateManager's metadata requirement)
        for name, module in [
            ("situation_encoder", self.encoder),
            ("memory_gate", self.gate),
            ("strategy_selector", self.selector),
            ("confidence_estimator", self.estimator),
        ]:
            path = state_dir / f"{name}.pt"
            tmp = path.with_suffix(".tmp")
            torch.save(module.state_dict(), tmp)
            tmp.rename(path)

        # Save conversation history
        hist_data = [
            {"role": t.role, "content": t.content, "timestamp": t.timestamp}
            for t in self.history[-100:]  # keep last 50 exchanges
        ]
        hist_path = state_dir / "conversation.json"
        tmp = hist_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(hist_data, f)
        tmp.rename(hist_path)

    def _load_state(self) -> None:
        """Load saved state if available."""
        state_dir = Path(self.config.state_dir)
        for name, module in [
            ("situation_encoder", self.encoder),
            ("memory_gate", self.gate),
            ("strategy_selector", self.selector),
            ("confidence_estimator", self.estimator),
        ]:
            path = state_dir / f"{name}.pt"
            if path.exists():
                try:
                    module.load_state_dict(torch.load(path, weights_only=True))
                except (RuntimeError, Exception):
                    pass  # Shape mismatch or corrupted, start fresh

        # Load conversation history
        hist_path = Path(self.config.state_dir) / "conversation.json"
        if hist_path.exists():
            try:
                with open(hist_path) as f:
                    data = json.load(f)
                self.history = [
                    ConversationTurn(
                        role=t["role"],
                        content=t["content"],
                        timestamp=t.get("timestamp", 0),
                    )
                    for t in data
                ]
            except (json.JSONDecodeError, KeyError):
                pass
