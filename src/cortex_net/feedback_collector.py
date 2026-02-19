"""Feedback Collector — extract training signal from implicit user behavior.

The hardest part of online learning: figuring out whether the agent did well
without explicit labels. This module infers reward signals from:

1. **Explicit feedback** — thumbs up/down, "thanks", "that's wrong"
2. **Behavioral signals** — user corrections, follow-up questions, conversation flow
3. **Engagement signals** — response time, conversation length, topic switches

Each signal maps to a reward in [-1, 1] that can update all four components.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


# --- Signal Patterns ---

POSITIVE_PATTERNS = [
    r"\bthanks?\b", r"\bthank you\b", r"\bperfect\b", r"\bexactly\b",
    r"\bthat('s| is) (right|correct|great|helpful|what i needed)\b",
    r"\bgot it\b", r"\bmakes sense\b", r"\bawesome\b", r"\bnice\b",
    r"👍", r"🎉", r"✅", r"💯",
]

NEGATIVE_PATTERNS = [
    r"\bthat('s| is) (\w+ )?(wrong|incorrect|not right|not what i)\b",
    r"\bno,?\s+(i meant|that's not|actually)\b",
    r"\byou('re| are) (wrong|confused|mistaken)\b",
    r"\bthat doesn't (help|work|make sense)\b",
    r"\bnever\s*mind\b", r"\bforget it\b",
    r"👎", r"❌", r"🤦",
]

CONFUSION_PATTERNS = [
    r"\bwhat\?+\b", r"\bhuh\?+\b", r"\bi don't understand\b",
    r"\bwhat do you mean\b", r"\bthat doesn't make sense\b",
    r"\bcan you (explain|clarify|rephrase)\b",
]

CORRECTION_PATTERNS = [
    r"\bno,\s", r"\bactually,?\s", r"\bi meant\b",
    r"\bnot that\b", r"\bthe other\b", r"\bi said\b",
]

_pos_compiled = [re.compile(p, re.IGNORECASE) for p in POSITIVE_PATTERNS]
_neg_compiled = [re.compile(p, re.IGNORECASE) for p in NEGATIVE_PATTERNS]
_conf_compiled = [re.compile(p, re.IGNORECASE) for p in CONFUSION_PATTERNS]
_corr_compiled = [re.compile(p, re.IGNORECASE) for p in CORRECTION_PATTERNS]


@dataclass
class FeedbackSignal:
    """A single feedback signal extracted from user behavior."""

    source: str  # "explicit", "behavioral", "engagement"
    signal_type: str  # "positive", "negative", "correction", "confusion", "neutral"
    strength: float  # [0, 1] how confident we are in this signal
    raw_text: str = ""
    timestamp: float = 0.0

    @property
    def reward(self) -> float:
        """Convert to reward in [-1, 1]."""
        if self.signal_type == "positive":
            return self.strength
        elif self.signal_type == "negative":
            return -self.strength
        elif self.signal_type == "correction":
            return -0.5 * self.strength  # milder than outright negative
        elif self.signal_type == "confusion":
            return -0.3 * self.strength  # agent wasn't clear
        return 0.0


@dataclass
class InteractionOutcome:
    """Full outcome of an interaction, combining all signals."""

    signals: list[FeedbackSignal] = field(default_factory=list)
    response_time_ms: float = 0.0
    conversation_continued: bool = True
    topic_switched: bool = False

    @property
    def reward(self) -> float:
        """Aggregate reward from all signals, clamped to [-1, 1]."""
        if not self.signals:
            # No explicit signal — use engagement heuristics
            reward = 0.0
            if self.conversation_continued:
                reward += 0.1  # user stayed, mild positive
            if self.topic_switched:
                reward -= 0.1  # topic switch might mean dissatisfaction
            return max(-1.0, min(1.0, reward))

        # Weight signals by strength and recency
        total = sum(s.reward for s in self.signals)
        return max(-1.0, min(1.0, total / len(self.signals)))

    @property
    def confidence_target(self) -> float:
        """Convert reward to confidence training target [0, 1]."""
        return (self.reward + 1.0) / 2.0  # map [-1,1] → [0,1]

    @property
    def memory_was_relevant(self) -> bool:
        """Infer whether retrieved memories were useful."""
        return self.reward > 0.0

    @property
    def strategy_was_correct(self) -> bool:
        """Infer whether the chosen strategy was appropriate."""
        return self.reward > -0.2  # generous — only clearly bad = wrong strategy


def extract_feedback(user_message: str) -> list[FeedbackSignal]:
    """Extract feedback signals from a user message."""
    signals = []
    text = user_message.strip()
    ts = time.time()

    # Check explicit positive
    pos_hits = sum(1 for p in _pos_compiled if p.search(text))
    if pos_hits > 0:
        signals.append(FeedbackSignal(
            source="explicit",
            signal_type="positive",
            strength=min(1.0, 0.5 + 0.2 * pos_hits),
            raw_text=text,
            timestamp=ts,
        ))

    # Check explicit negative
    neg_hits = sum(1 for p in _neg_compiled if p.search(text))
    if neg_hits > 0:
        signals.append(FeedbackSignal(
            source="explicit",
            signal_type="negative",
            strength=min(1.0, 0.5 + 0.2 * neg_hits),
            raw_text=text,
            timestamp=ts,
        ))

    # Check correction
    corr_hits = sum(1 for p in _corr_compiled if p.search(text))
    if corr_hits > 0:
        signals.append(FeedbackSignal(
            source="behavioral",
            signal_type="correction",
            strength=min(1.0, 0.4 + 0.2 * corr_hits),
            raw_text=text,
            timestamp=ts,
        ))

    # Check confusion
    conf_hits = sum(1 for p in _conf_compiled if p.search(text))
    if conf_hits > 0:
        signals.append(FeedbackSignal(
            source="behavioral",
            signal_type="confusion",
            strength=min(1.0, 0.4 + 0.2 * conf_hits),
            raw_text=text,
            timestamp=ts,
        ))

    return signals


@dataclass
class ReplayEntry:
    """A stored experience for replay-based online training."""

    # Situation
    situation_emb: list[float]  # stored as list for JSON serialization

    # What was retrieved/selected
    memory_indices: list[int]
    memory_scores: list[float]
    strategy_id: str
    confidence: float

    # Outcome
    reward: float
    confidence_target: float
    memory_was_relevant: bool
    strategy_was_correct: bool

    # Metadata
    timestamp: float = 0.0
    query: str = ""


class ReplayBuffer:
    """Experience replay buffer — stores interactions for online training.

    Maintains a fixed-size buffer of past interactions with outcomes.
    Supports sampling for mini-batch training and persistence to disk.

    Args:
        capacity: Maximum entries to store.
        path: Path to persist buffer to disk.
    """

    def __init__(self, capacity: int = 10000, path: str | Path = "./state/replay_buffer.jsonl") -> None:
        self.capacity = capacity
        self.path = Path(path)
        self.entries: list[ReplayEntry] = []
        self._load()

    def add(self, entry: ReplayEntry) -> None:
        """Add an experience to the buffer."""
        self.entries.append(entry)
        if len(self.entries) > self.capacity:
            # Remove oldest
            self.entries = self.entries[-self.capacity:]
        self._save()

    def sample(self, n: int, prioritize_recent: bool = True) -> list[ReplayEntry]:
        """Sample n entries from the buffer.

        Args:
            n: Number of entries to sample.
            prioritize_recent: Weight recent entries higher.

        Returns:
            List of sampled entries.
        """
        if not self.entries:
            return []
        n = min(n, len(self.entries))

        if prioritize_recent:
            # Exponential recency weighting
            weights = torch.exp(torch.linspace(-2, 0, len(self.entries)))
            weights = weights / weights.sum()
            indices = torch.multinomial(weights, n, replacement=False)
            return [self.entries[i] for i in indices.tolist()]
        else:
            indices = torch.randperm(len(self.entries))[:n]
            return [self.entries[i] for i in indices.tolist()]

    def __len__(self) -> int:
        return len(self.entries)

    def _save(self) -> None:
        """Persist buffer to disk (atomic write)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for entry in self.entries:
                f.write(json.dumps(entry.__dict__) + "\n")
        tmp.rename(self.path)

    def _load(self) -> None:
        """Load buffer from disk."""
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self.entries.append(ReplayEntry(**data))
        except (json.JSONDecodeError, TypeError):
            self.entries = []  # corrupted, start fresh
