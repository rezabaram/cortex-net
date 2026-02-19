"""Agent monitoring — structured logging for observability.

Logs every interaction with full context: what was retrieved, which strategy
was picked, confidence score, feedback signals, timing, and learning updates.

Output: JSONL file + optional human-readable log.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

log = logging.getLogger("cortex-monitor")


@dataclass
class InteractionLog:
    """Complete record of one agent interaction."""

    # Identity
    timestamp: float = 0.0
    turn_number: int = 0

    # Input
    user_message: str = ""
    message_length: int = 0

    # Situation
    situation_norm: float = 0.0

    # Memory retrieval
    memories_retrieved: int = 0
    memory_scores: list[float] = field(default_factory=list)
    memory_texts: list[str] = field(default_factory=list)
    top_memory_score: float = 0.0
    mean_memory_score: float = 0.0

    # Strategy
    strategy_id: str = ""
    strategy_confidence: float = 0.0
    strategy_probabilities: dict[str, float] = field(default_factory=dict)

    # Confidence
    confidence: float = 0.0
    confidence_action: str = ""  # proceed / hedge / escalate

    # Conversation Gate
    history_turns_total: int = 0
    history_turns_selected: int = 0
    conv_gate_threshold: float = 0.0

    # Context assembly
    context_messages: int = 0         # messages sent to LLM
    context_chars: int = 0            # total chars in prompt (system + messages)
    system_prompt_chars: int = 0      # system prompt size

    # LLM
    llm_model: str = ""
    llm_input_tokens: int = 0         # from API response usage
    llm_output_tokens: int = 0
    llm_total_tokens: int = 0

    # Response
    response_length: int = 0
    tool_calls: int = 0
    tool_names: list[str] = field(default_factory=list)

    # Feedback (from PREVIOUS turn)
    feedback_reward: float | None = None
    feedback_type: str = ""  # positive / negative / correction / neutral

    # Learning
    online_update: bool = False
    ema_loss: float = 0.0
    buffer_size: int = 0

    # Timing
    encoding_ms: float = 0.0
    retrieval_ms: float = 0.0
    strategy_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0

    # Errors
    error: str = ""


class AgentMonitor:
    """Collects and persists agent interaction logs.

    Writes to:
    - JSONL file (machine-readable, for analysis)
    - Python logger (human-readable, for journalctl)
    """

    def __init__(self, log_dir: str | Path = "./state", name: str = "atlas") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / f"{name}_interactions.jsonl"
        self.name = name
        self._turn_count = 0

        # Summary stats
        self._total_interactions = 0
        self._total_positive = 0
        self._total_negative = 0
        self._total_corrections = 0
        self._total_tool_calls = 0
        self._avg_confidence = 0.0
        self._avg_response_time_ms = 0.0

        # Load existing count
        if self.jsonl_path.exists():
            try:
                self._total_interactions = sum(1 for _ in open(self.jsonl_path))
                self._turn_count = self._total_interactions
            except Exception:
                pass

    def new_interaction(self) -> InteractionLog:
        """Create a new interaction log entry."""
        self._turn_count += 1
        return InteractionLog(
            timestamp=time.time(),
            turn_number=self._turn_count,
        )

    def record(self, entry: InteractionLog) -> None:
        """Persist an interaction log entry."""
        self._total_interactions += 1

        # Update summary stats
        if entry.feedback_type == "positive":
            self._total_positive += 1
        elif entry.feedback_type == "negative":
            self._total_negative += 1
        elif entry.feedback_type == "correction":
            self._total_corrections += 1
        self._total_tool_calls += entry.tool_calls

        # Running average
        n = self._total_interactions
        self._avg_confidence = (self._avg_confidence * (n - 1) + entry.confidence) / n
        self._avg_response_time_ms = (self._avg_response_time_ms * (n - 1) + entry.total_ms) / n

        # Write JSONL
        try:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(asdict(entry), default=str) + "\n")
        except Exception as e:
            log.error(f"Failed to write interaction log: {e}")

        # Human-readable log
        mem_info = f"mem={entry.memories_retrieved}(top={entry.top_memory_score:.2f})" if entry.memories_retrieved else "mem=0"
        fb_info = f"fb={entry.feedback_type}({entry.feedback_reward:.2f})" if entry.feedback_reward is not None else ""
        tool_info = f"tools={entry.tool_calls}" if entry.tool_calls else ""
        learn_info = "📚 UPDATED" if entry.online_update else ""

        ctx_info = f"ctx={entry.context_messages}msg/{entry.context_chars}ch"
        token_info = f"tok={entry.llm_input_tokens}→{entry.llm_output_tokens}" if entry.llm_input_tokens else ""
        gate_info = f"gate={entry.history_turns_selected}/{entry.history_turns_total}" if entry.history_turns_total else ""

        parts = [
            f"[turn {entry.turn_number}]",
            f"strat={entry.strategy_id}",
            f"conf={entry.confidence:.2f}",
            mem_info,
            gate_info,
            ctx_info,
            token_info,
            f"resp={entry.response_length}ch",
            f"{entry.total_ms:.0f}ms",
            tool_info,
            fb_info,
            learn_info,
        ]
        log.info(" ".join(p for p in parts if p))

        if entry.error:
            log.error(f"[turn {entry.turn_number}] Error: {entry.error}")

    def summary(self) -> dict[str, Any]:
        """Get monitoring summary."""
        return {
            "total_interactions": self._total_interactions,
            "positive_feedback": self._total_positive,
            "negative_feedback": self._total_negative,
            "corrections": self._total_corrections,
            "total_tool_calls": self._total_tool_calls,
            "avg_confidence": round(self._avg_confidence, 3),
            "avg_response_time_ms": round(self._avg_response_time_ms, 0),
        }

    def summary_text(self) -> str:
        """Human-readable summary."""
        s = self.summary()
        return (
            f"📊 {self.name} — {s['total_interactions']} interactions | "
            f"👍 {s['positive_feedback']} 👎 {s['negative_feedback']} ✏️ {s['corrections']} | "
            f"conf={s['avg_confidence']:.2f} | "
            f"⏱️ {s['avg_response_time_ms']:.0f}ms avg | "
            f"🔧 {s['total_tool_calls']} tool calls"
        )
