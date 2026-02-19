"""Online Trainer — continuous learning from real interactions.

Connects the feedback loop:
  Agent acts → User reacts → Extract signal → Update components → Agent improves

Uses experience replay to stabilize training and avoid catastrophic forgetting.
Updates are small and incremental — no full retraining.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from cortex_net.confidence_estimator import (
    CalibrationLoss,
    ConfidenceEstimator,
    ContextSummary,
)
from cortex_net.feedback_collector import (
    InteractionOutcome,
    ReplayBuffer,
    ReplayEntry,
    extract_feedback,
)
from cortex_net.memory_gate import MemoryGate
from cortex_net.situation_encoder import SituationEncoder
from cortex_net.strategy_selector import StrategyRegistry, StrategySelector


@dataclass
class OnlineUpdate:
    """Result of an online training step."""

    num_samples: int
    memory_loss: float = 0.0
    strategy_loss: float = 0.0
    confidence_loss: float = 0.0
    total_loss: float = 0.0


class OnlineTrainer:
    """Continuously updates cortex-net components from real interactions.

    Design choices:
    - Small learning rate (1e-4) — incremental updates, not revolution
    - Experience replay — sample from buffer to avoid recency bias
    - EMA tracking — detect if online learning is degrading performance
    - Update frequency — train every N interactions, not every single one

    Args:
        encoder: Situation Encoder.
        gate: Memory Gate.
        selector: Strategy Selector.
        estimator: Confidence Estimator.
        registry: Strategy profile registry.
        buffer: Experience replay buffer.
        lr: Learning rate (small for stability).
        batch_size: Samples per update step.
        update_every: Train after this many new interactions.
        memory_weight: Loss weight for memory gate.
        strategy_weight: Loss weight for strategy selector.
        confidence_weight: Loss weight for confidence estimator.
        max_grad_norm: Gradient clipping.
    """

    def __init__(
        self,
        encoder: SituationEncoder,
        gate: MemoryGate,
        selector: StrategySelector,
        estimator: ConfidenceEstimator,
        registry: StrategyRegistry = None,
        *,
        conversation_gate=None,
        buffer: ReplayBuffer | None = None,
        lr: float = 1e-4,
        batch_size: int = 16,
        update_every: int = 5,
        memory_weight: float = 1.0,
        strategy_weight: float = 1.0,
        confidence_weight: float = 0.5,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.encoder = encoder
        self.gate = gate
        self.selector = selector
        self.estimator = estimator
        self.conversation_gate = conversation_gate
        self.registry = registry or StrategyRegistry()
        self.buffer = buffer or ReplayBuffer()

        self.batch_size = batch_size
        self.update_every = update_every
        self.memory_weight = memory_weight
        self.strategy_weight = strategy_weight
        self.confidence_weight = confidence_weight
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(
            list(encoder.parameters())
            + list(gate.parameters())
            + list(selector.parameters())
            + list(estimator.parameters())
            + (list(conversation_gate.parameters()) if conversation_gate else []),
            lr=lr,
        )

        self.cal_loss = CalibrationLoss()
        self._interactions_since_update = 0
        self._total_updates = 0

        # EMA loss tracking for degradation detection
        self._ema_loss = 0.0
        self._ema_alpha = 0.1

    def record_interaction(
        self,
        situation_emb: torch.Tensor,
        memory_indices: list[int],
        memory_scores: list[float],
        strategy_id: str,
        confidence: float,
        user_response: str,
        query: str = "",
        conversation_continued: bool = True,
        topic_switched: bool = False,
    ) -> InteractionOutcome:
        """Record an interaction and its outcome.

        Call this after the agent responds and the user reacts.

        Args:
            situation_emb: The situation embedding used for this interaction.
            memory_indices: Indices of memories that were retrieved.
            memory_scores: Scores of retrieved memories.
            strategy_id: Strategy that was selected.
            confidence: Confidence score that was predicted.
            user_response: The user's next message (for feedback extraction).
            query: Original query (for logging).
            conversation_continued: Whether the user sent another message.
            topic_switched: Whether the user changed topics.

        Returns:
            The extracted outcome.
        """
        # Extract feedback signals
        signals = extract_feedback(user_response)
        outcome = InteractionOutcome(
            signals=signals,
            conversation_continued=conversation_continued,
            topic_switched=topic_switched,
        )

        # Store in replay buffer
        entry = ReplayEntry(
            situation_emb=situation_emb.detach().cpu().tolist(),
            memory_indices=memory_indices,
            memory_scores=memory_scores,
            strategy_id=strategy_id,
            confidence=confidence,
            reward=outcome.reward,
            confidence_target=outcome.confidence_target,
            memory_was_relevant=outcome.memory_was_relevant,
            strategy_was_correct=outcome.strategy_was_correct,
            query=query,
        )
        self.buffer.add(entry)

        # Maybe update
        self._interactions_since_update += 1
        update = None
        if self._interactions_since_update >= self.update_every:
            update = self.step()
            self._interactions_since_update = 0

        return outcome

    def step(self) -> OnlineUpdate:
        """Run one training step using replay buffer samples.

        Returns:
            Training metrics for this step.
        """
        if len(self.buffer) < 2:
            return OnlineUpdate(num_samples=0)

        device = next(self.encoder.parameters()).device
        batch = self.buffer.sample(self.batch_size)

        total_mem = total_strat = total_conf = 0.0
        count = 0

        for entry in batch:
            sit = torch.tensor(entry.situation_emb, device=device)
            loss = torch.tensor(0.0, device=device)

            # Memory Gate: use reward as relevance signal
            # If reward > 0, the retrieved memories were useful → reinforce
            # If reward < 0, they weren't → push away
            if entry.memory_scores and entry.reward != 0:
                # Create pseudo positive/negative from reward direction
                score_tensor = torch.tensor(entry.memory_scores, device=device)
                if entry.memory_was_relevant:
                    # Reinforce: push situation toward retrieved memory directions
                    # This is a simplified signal — full version would need memory embeddings
                    target_conf = torch.tensor(1.0, device=device)
                else:
                    target_conf = torch.tensor(0.0, device=device)

                pred = torch.sigmoid(score_tensor.mean())
                mem_loss = nn.functional.binary_cross_entropy(pred, target_conf)
                loss = loss + self.memory_weight * mem_loss
                total_mem += mem_loss.item()

            # Strategy Selector: if strategy was wrong, train toward ... what?
            # We don't know the right strategy from implicit feedback alone.
            # But we can penalize the chosen strategy if reward was negative.
            if entry.strategy_id and not entry.strategy_was_correct:
                idx = self.registry.id_to_index(entry.strategy_id)
                if idx is not None:
                    # Negative reward: increase loss for chosen strategy
                    strat_loss = self.selector.strategy_loss(sit, idx)
                    # Invert: we want to push AWAY from this strategy
                    loss = loss + self.strategy_weight * (-strat_loss * 0.1)
                    total_strat += strat_loss.item()

            # Confidence Estimator: train toward actual outcome
            ctx_tensor = ContextSummary(
                top_memory_score=entry.memory_scores[0] if entry.memory_scores else 0,
                mean_memory_score=sum(entry.memory_scores) / len(entry.memory_scores) if entry.memory_scores else 0,
            ).to_tensor().to(device)

            pred_conf = self.estimator(sit, ctx_tensor)
            target = torch.tensor(entry.confidence_target, device=device)
            conf_loss = self.cal_loss(pred_conf.unsqueeze(0), target.unsqueeze(0))
            loss = loss + self.confidence_weight * conf_loss
            total_conf += conf_loss.item()

            if loss.requires_grad:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters())
                    + list(self.gate.parameters())
                    + list(self.selector.parameters())
                    + list(self.estimator.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

            count += 1

        self._total_updates += 1
        total = (total_mem + total_strat + total_conf) / max(count, 1)

        # EMA tracking
        self._ema_loss = self._ema_alpha * total + (1 - self._ema_alpha) * self._ema_loss

        return OnlineUpdate(
            num_samples=count,
            memory_loss=total_mem / max(count, 1),
            strategy_loss=total_strat / max(count, 1),
            confidence_loss=total_conf / max(count, 1),
            total_loss=total,
        )

    @property
    def ema_loss(self) -> float:
        """Exponential moving average of training loss."""
        return self._ema_loss

    @property
    def total_updates(self) -> int:
        return self._total_updates

    @property
    def buffer_size(self) -> int:
        return len(self.buffer)
