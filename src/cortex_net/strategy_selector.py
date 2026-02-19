"""Strategy Selector — learns which approach works for which situation.

Classifies situations into strategy profiles. Each profile represents a
different way the agent can respond: deep research, quick answer, clarify
first, proactive suggest, hedge & escalate, etc.

Architecture: linear classification head on situation embeddings → softmax
over strategy profiles.

Training signal: which strategy led to task success in similar past situations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Strategy Profiles ────────────────────────────────────────────────

@dataclass
class StrategyProfile:
    """A named strategy with its configuration."""

    id: str
    name: str
    description: str
    prompt_framing: str = ""
    reasoning_style: str = "direct"  # "direct" | "step_by_step" | "exploratory"
    response_format: str = "concise"  # "concise" | "thorough" | "structured"
    tool_permissions: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


# Default strategy profiles
DEFAULT_STRATEGIES = [
    StrategyProfile(
        id="deep_research",
        name="Deep Research",
        description="Complex factual question needing thorough analysis",
        prompt_framing="Think step by step. Cite sources. Be thorough.",
        reasoning_style="step_by_step",
        response_format="thorough",
        tool_permissions=["web_search", "code_exec"],
    ),
    StrategyProfile(
        id="quick_answer",
        name="Quick Answer",
        description="Simple known question, respond concisely",
        prompt_framing="Be concise and direct. No unnecessary explanation.",
        reasoning_style="direct",
        response_format="concise",
    ),
    StrategyProfile(
        id="clarify_first",
        name="Clarify First",
        description="Ambiguous request, ask one clarifying question",
        prompt_framing="Ask one specific clarifying question before proceeding.",
        reasoning_style="direct",
        response_format="concise",
    ),
    StrategyProfile(
        id="proactive_suggest",
        name="Proactive Suggest",
        description="Pattern detected, surface insight and suggest action",
        prompt_framing="Surface the insight you've noticed. Suggest a concrete action.",
        reasoning_style="exploratory",
        response_format="structured",
    ),
    StrategyProfile(
        id="hedge_escalate",
        name="Hedge & Escalate",
        description="Low confidence, state uncertainty and offer alternatives",
        prompt_framing="State what you're unsure about. Offer alternatives or suggest who to ask.",
        reasoning_style="direct",
        response_format="concise",
    ),
    StrategyProfile(
        id="step_by_step",
        name="Step by Step",
        description="Multi-step task needing structured execution",
        prompt_framing="Break this into numbered steps. Execute each step carefully.",
        reasoning_style="step_by_step",
        response_format="structured",
        tool_permissions=["code_exec"],
    ),
    StrategyProfile(
        id="creative",
        name="Creative",
        description="Open-ended creative task, explore possibilities",
        prompt_framing="Explore multiple angles. Be creative and generative.",
        reasoning_style="exploratory",
        response_format="thorough",
    ),
    StrategyProfile(
        id="code_assist",
        name="Code Assist",
        description="Programming help, show code with explanation",
        prompt_framing="Show working code. Explain key decisions briefly.",
        reasoning_style="step_by_step",
        response_format="structured",
        tool_permissions=["code_exec"],
    ),
    StrategyProfile(
        id="summarize",
        name="Summarize",
        description="Distill information into key points",
        prompt_framing="Extract the key points. Be brief and actionable.",
        reasoning_style="direct",
        response_format="structured",
    ),
    StrategyProfile(
        id="empathize",
        name="Empathize & Support",
        description="User expressing frustration or confusion, be supportive",
        prompt_framing="Acknowledge the difficulty. Be supportive. Then help.",
        reasoning_style="direct",
        response_format="concise",
    ),
]


class StrategyRegistry:
    """Registry of available strategy profiles."""

    def __init__(self, strategies: list[StrategyProfile] | None = None) -> None:
        self.strategies = strategies or list(DEFAULT_STRATEGIES)
        self._id_to_idx = {s.id: i for i, s in enumerate(self.strategies)}

    def __len__(self) -> int:
        return len(self.strategies)

    def __getitem__(self, idx: int) -> StrategyProfile:
        return self.strategies[idx]

    def get_by_id(self, strategy_id: str) -> StrategyProfile | None:
        idx = self._id_to_idx.get(strategy_id)
        return self.strategies[idx] if idx is not None else None

    def id_to_index(self, strategy_id: str) -> int | None:
        return self._id_to_idx.get(strategy_id)

    @property
    def ids(self) -> list[str]:
        return [s.id for s in self.strategies]


# ── Strategy Selector Model ──────────────────────────────────────────

@dataclass
class StrategySelection:
    """Result of strategy selection."""

    strategy_id: str
    strategy: StrategyProfile
    confidence: float
    probabilities: dict[str, float]


class StrategySelector(nn.Module):
    """Learned strategy selection from situation embeddings.

    Architecture:
        situation_embedding → Linear → ReLU → Linear → softmax → strategy distribution

    Supports:
    - Hard selection (argmax)
    - Soft selection (probability distribution)
    - Exploration (epsilon-greedy or temperature sampling)

    Args:
        situation_dim: Dimension of input situation embeddings.
        num_strategies: Number of strategy profiles.
        hidden_dim: Hidden layer dimension.
        exploration_rate: Probability of random strategy (epsilon-greedy).
        temperature: Softmax temperature (higher = more exploration).
    """

    def __init__(
        self,
        situation_dim: int = 256,
        num_strategies: int = 10,
        hidden_dim: int = 128,
        exploration_rate: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.situation_dim = situation_dim
        self.num_strategies = num_strategies
        self.exploration_rate = exploration_rate
        self.temperature = temperature

        self.classifier = nn.Sequential(
            nn.Linear(situation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_strategies),
        )

        # Track per-strategy usage counts for diversity metrics
        self.register_buffer(
            "_usage_counts",
            torch.zeros(num_strategies, dtype=torch.long),
        )

    def forward(self, situation: torch.Tensor) -> torch.Tensor:
        """Compute strategy logits.

        Args:
            situation: (D,) or (B, D) situation embedding.

        Returns:
            Logits: (num_strategies,) or (B, num_strategies).
        """
        return self.classifier(situation)

    def select(
        self,
        situation: torch.Tensor,
        registry: StrategyRegistry,
        explore: bool = True,
    ) -> StrategySelection:
        """Select a strategy for the given situation.

        Args:
            situation: (D,) situation embedding.
            registry: Strategy registry.
            explore: If True, occasionally pick random strategies.

        Returns:
            StrategySelection with chosen strategy and probabilities.
        """
        with torch.no_grad():
            logits = self.forward(situation)
            probs = F.softmax(logits / self.temperature, dim=-1)

            # Epsilon-greedy exploration
            if explore and torch.rand(1).item() < self.exploration_rate:
                idx = torch.randint(0, self.num_strategies, (1,)).item()
            else:
                idx = torch.argmax(probs).item()

            self._usage_counts[idx] += 1

        strategy = registry[idx]
        prob_dict = {
            registry[i].id: probs[i].item()
            for i in range(min(len(registry), self.num_strategies))
        }

        return StrategySelection(
            strategy_id=strategy.id,
            strategy=strategy,
            confidence=probs[idx].item(),
            probabilities=prob_dict,
        )

    def strategy_loss(
        self,
        situation: torch.Tensor,
        target_strategy_idx: int,
    ) -> torch.Tensor:
        """Cross-entropy loss for supervised strategy training.

        Args:
            situation: (D,) situation embedding.
            target_strategy_idx: Index of the correct strategy.

        Returns:
            Scalar loss.
        """
        logits = self.forward(situation.unsqueeze(0))
        target = torch.tensor([target_strategy_idx], device=situation.device)
        return F.cross_entropy(logits, target)

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        situation: torch.Tensor,
        target_strategy_idx: int,
    ) -> float:
        """Single training step.

        Args:
            optimizer: Optimizer for this module's parameters.
            situation: (D,) situation embedding.
            target_strategy_idx: Ground truth strategy index.

        Returns:
            Loss value as float.
        """
        optimizer.zero_grad()
        loss = self.strategy_loss(situation, target_strategy_idx)
        loss.backward()
        optimizer.step()
        return loss.item()

    def diversity_score(self) -> float:
        """Measure strategy usage diversity (0 = one strategy only, 1 = uniform).

        Uses normalized entropy of usage distribution.
        """
        counts = self._usage_counts.float()
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]  # filter zeros for log
        entropy = -(probs * probs.log()).sum()
        max_entropy = torch.tensor(float(self.num_strategies)).log()
        return (entropy / max_entropy).item() if max_entropy > 0 else 0.0

    def usage_summary(self, registry: StrategyRegistry) -> dict[str, int]:
        """Return usage counts per strategy."""
        return {
            registry[i].id: self._usage_counts[i].item()
            for i in range(min(len(registry), self.num_strategies))
        }
