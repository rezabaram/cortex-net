"""Joint Trainer — end-to-end training of all cortex-net components.

Trains all four components with shared gradients through the Situation Encoder:
- Memory Gate: contrastive loss on relevant/irrelevant memories
- Strategy Selector: cross-entropy on labeled strategy
- Confidence Estimator: calibration loss on predicted vs actual outcome

The Situation Encoder gets gradient signal from all three, learning a
representation that serves all downstream tasks simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortex_net.confidence_estimator import (
    CalibrationLoss,
    ConfidenceEstimator,
    ContextSummary,
)
from cortex_net.memory_gate import MemoryGate
from cortex_net.situation_encoder import SituationEncoder, extract_metadata_features
from cortex_net.strategy_selector import StrategyRegistry, StrategySelector


@dataclass
class TrainingSample:
    """A single training example with labels for all components."""

    # Situation
    message_emb: torch.Tensor  # (D,)
    history_emb: torch.Tensor  # (D,)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Memory Gate labels
    memory_embs: torch.Tensor | None = None  # (N, D)
    relevant_indices: set[int] = field(default_factory=set)

    # Strategy label
    strategy_id: str = ""

    # Confidence label
    outcome: float = 0.5  # 0.0 = wrong, 1.0 = correct


@dataclass
class JointTrainingMetrics:
    """Metrics from joint training."""

    epochs: int = 0
    total_loss: list[float] = field(default_factory=list)
    memory_loss: list[float] = field(default_factory=list)
    strategy_loss: list[float] = field(default_factory=list)
    confidence_loss: list[float] = field(default_factory=list)

    def summary(self) -> str:
        if not self.total_loss:
            return "No training data"
        return (
            f"Joint Training: {self.epochs} epochs\n"
            f"  Total loss:      {self.total_loss[0]:.4f} → {self.total_loss[-1]:.4f}\n"
            f"  Memory loss:     {self.memory_loss[0]:.4f} → {self.memory_loss[-1]:.4f}\n"
            f"  Strategy loss:   {self.strategy_loss[0]:.4f} → {self.strategy_loss[-1]:.4f}\n"
            f"  Confidence loss: {self.confidence_loss[0]:.4f} → {self.confidence_loss[-1]:.4f}"
        )


class JointTrainer:
    """Trains all cortex-net components end-to-end.

    All gradients flow through the Situation Encoder, so it learns a
    representation that serves memory retrieval, strategy selection,
    and confidence estimation simultaneously.

    Args:
        encoder: Situation Encoder.
        gate: Memory Gate.
        selector: Strategy Selector.
        estimator: Confidence Estimator.
        registry: Strategy profile registry.
        lr: Learning rate.
        memory_weight: Weight for memory gate loss.
        strategy_weight: Weight for strategy selector loss.
        confidence_weight: Weight for confidence estimator loss.
        max_grad_norm: Gradient clipping norm.
    """

    def __init__(
        self,
        encoder: SituationEncoder,
        gate: MemoryGate,
        selector: StrategySelector,
        estimator: ConfidenceEstimator,
        registry: StrategyRegistry,
        lr: float = 1e-3,
        memory_weight: float = 1.0,
        strategy_weight: float = 1.0,
        confidence_weight: float = 0.5,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.encoder = encoder
        self.gate = gate
        self.selector = selector
        self.estimator = estimator
        self.registry = registry

        self.memory_weight = memory_weight
        self.strategy_weight = strategy_weight
        self.confidence_weight = confidence_weight
        self.max_grad_norm = max_grad_norm

        # Single optimizer for all components
        all_params = (
            list(encoder.parameters())
            + list(gate.parameters())
            + list(selector.parameters())
            + list(estimator.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr * 0.1
        )

        self.cal_loss = CalibrationLoss()

    def train(
        self,
        samples: list[TrainingSample],
        epochs: int = 100,
    ) -> JointTrainingMetrics:
        """Train all components jointly.

        Args:
            samples: Training examples with labels for all components.
            epochs: Number of training epochs.

        Returns:
            Training metrics.
        """
        device = next(self.encoder.parameters()).device
        metrics = JointTrainingMetrics(epochs=epochs)

        for epoch in range(epochs):
            epoch_total = 0.0
            epoch_mem = 0.0
            epoch_strat = 0.0
            epoch_conf = 0.0
            count = 0

            for sample in samples:
                meta = extract_metadata_features(
                    sample.metadata
                ).to(device)

                # Forward through Situation Encoder (gradients flow back)
                situation = self.encoder(
                    sample.message_emb.to(device),
                    sample.history_emb.to(device),
                    meta,
                )

                loss = torch.tensor(0.0, device=device)

                # Memory Gate loss
                if sample.memory_embs is not None and sample.relevant_indices:
                    mem_embs = sample.memory_embs.to(device)
                    pos_idx = list(sample.relevant_indices)
                    neg_idx = [i for i in range(mem_embs.shape[0]) if i not in sample.relevant_indices]
                    if pos_idx and neg_idx:
                        mem_loss = self.gate.contrastive_loss(
                            situation, mem_embs[pos_idx], mem_embs[neg_idx]
                        )
                        loss = loss + self.memory_weight * mem_loss
                        epoch_mem += mem_loss.item()

                # Strategy Selector loss
                if sample.strategy_id:
                    target_idx = self.registry.id_to_index(sample.strategy_id)
                    if target_idx is not None:
                        strat_loss = self.selector.strategy_loss(situation, target_idx)
                        loss = loss + self.strategy_weight * strat_loss
                        epoch_strat += strat_loss.item()

                # Confidence Estimator loss
                ctx_summary = ContextSummary()
                if sample.memory_embs is not None:
                    with torch.no_grad():
                        _, scores = self.gate.select_top_k(
                            situation.detach(), sample.memory_embs.to(device), k=3
                        )
                    ctx_summary.top_memory_score = scores[0].item() if len(scores) > 0 else 0
                    ctx_summary.mean_memory_score = scores.mean().item() if len(scores) > 0 else 0

                ctx_tensor = ctx_summary.to_tensor().to(device)
                pred_conf = self.estimator(situation, ctx_tensor)
                actual = torch.tensor(sample.outcome, device=device)
                conf_loss = self.cal_loss(pred_conf.unsqueeze(0), actual.unsqueeze(0))
                loss = loss + self.confidence_weight * conf_loss
                epoch_conf += conf_loss.item()

                # Backward
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters())
                    + list(self.gate.parameters())
                    + list(self.selector.parameters())
                    + list(self.estimator.parameters()),
                    self.max_grad_norm,
                )

                self.optimizer.step()
                epoch_total += loss.item()
                count += 1

            self.scheduler.step()

            if count > 0:
                metrics.total_loss.append(epoch_total / count)
                metrics.memory_loss.append(epoch_mem / count)
                metrics.strategy_loss.append(epoch_strat / count)
                metrics.confidence_loss.append(epoch_conf / count)

        # Mark gate as trained
        self.gate._trained_flag.fill_(1)

        return metrics


@dataclass
class AblationResult:
    """Result of an ablation study."""

    full_system: dict[str, float]
    without_encoder: dict[str, float]
    without_gate: dict[str, float]
    without_strategy: dict[str, float]
    without_confidence: dict[str, float]

    def summary(self) -> str:
        lines = ["Ablation Study", ""]
        for label, metrics in [
            ("Full system", self.full_system),
            ("w/o Situation Encoder", self.without_encoder),
            ("w/o Memory Gate (cosine)", self.without_gate),
            ("w/o Strategy Selector", self.without_strategy),
            ("w/o Confidence Estimator", self.without_confidence),
        ]:
            parts = ", ".join(
                f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in metrics.items()
            )
            lines.append(f"  {label}: {parts}")
        return "\n".join(lines)
