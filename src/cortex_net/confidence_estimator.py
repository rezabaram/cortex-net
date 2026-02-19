"""Confidence Estimator — predicts how confident the system should be.

Trained on calibration against actual outcomes. When it says 70% confident,
it should be right ~70% of the time. Low confidence triggers hedging,
clarification requests, or escalation.

Architecture: 2-layer MLP → scalar output [0, 1]

Inputs:
    - Situation embedding
    - Context summary (retrieval quality signals)

Training: calibration loss — penalizes overconfidence and underconfidence equally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ContextSummary:
    """Summary of assembled context quality — fed to the Confidence Estimator.

    These are signals about how good the retrieved context is,
    not the content itself.
    """

    num_memories_retrieved: int = 0
    top_memory_score: float = 0.0
    mean_memory_score: float = 0.0
    score_spread: float = 0.0  # max - min score (high spread = clear winner)
    strategy_confidence: float = 0.0  # from StrategySelector
    query_length: int = 0
    history_length: int = 0

    def to_tensor(self) -> torch.Tensor:
        """Convert to normalized feature vector."""
        return torch.tensor([
            min(self.num_memories_retrieved / 10.0, 1.0),
            self.top_memory_score,  # already ~[0,1] for cosine
            self.mean_memory_score,
            self.score_spread,
            self.strategy_confidence,
            min(self.query_length / 500.0, 1.0),
            min(self.history_length / 50.0, 1.0),
        ], dtype=torch.float32)


CONTEXT_SUMMARY_DIM = 7  # matches ContextSummary.to_tensor() output


@dataclass
class ConfidenceResult:
    """Result of confidence estimation."""

    confidence: float  # 0.0 - 1.0
    action: str  # "proceed" | "hedge" | "escalate"
    raw_logit: float = 0.0
    entropy: float = 0.0  # entropy of confidence distribution
    is_certain: bool = False  # True if entropy below threshold
    fallback: str = ""  # "clarify" when is_certain is False

    # Entropy threshold for certainty (in nats)
    ENTROPY_THRESHOLD: float = 0.5

    @staticmethod
    def action_from_confidence(confidence: float) -> str:
        if confidence >= 0.8:
            return "proceed"
        elif confidence >= 0.4:
            return "hedge"
        else:
            return "escalate"

    @staticmethod
    def compute_entropy(confidence: float) -> float:
        """Compute binary entropy for a confidence value.
        
        For binary outcome (correct/incorrect), entropy is maximized
        at p=0.5 and minimized at p=0 or p=1.
        """
        # Clamp to avoid log(0)
        p = max(1e-7, min(1 - 1e-7, confidence))
        q = 1 - p
        return -(p * torch.log(torch.tensor(p)) + q * torch.log(torch.tensor(q))).item()

    @staticmethod
    def determine_fallback(is_certain: bool, action: str) -> str:
        """Determine fallback strategy when uncertain."""
        if not is_certain:
            return "clarify"
        # If certain but action is escalate, suggest human review
        if action == "escalate":
            return "human_review"
        return ""

    @staticmethod
    def from_confidence(confidence: float) -> "ConfidenceResult":
        """Create ConfidenceResult with entropy, certainty, and fallback."""
        entropy = ConfidenceResult.compute_entropy(confidence)
        is_certain = entropy < ConfidenceResult.ENTROPY_THRESHOLD
        action = ConfidenceResult.action_from_confidence(confidence)
        fallback = ConfidenceResult.determine_fallback(is_certain, action)
        
        # Map confidence to raw_logit (inverse sigmoid)
        import math
        raw_logit = math.log(confidence / (1 - confidence + 1e-7)) if confidence < 0.99 else 5.0
        
        return ConfidenceResult(
            confidence=confidence,
            action=action,
            raw_logit=raw_logit,
            entropy=entropy,
            is_certain=is_certain,
            fallback=fallback,
        )


class ConfidenceEstimator(nn.Module):
    """Learned confidence estimation.

    Takes situation embedding + context quality summary and predicts
    a calibrated confidence score.

    Calibration means: if it outputs 0.7, the system should be correct
    ~70% of the time. This is enforced via calibration loss during training.

    Args:
        situation_dim: Dimension of situation embeddings.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        situation_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.situation_dim = situation_dim

        input_dim = situation_dim + CONTEXT_SUMMARY_DIM

        self.estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        situation: torch.Tensor,
        context_summary: torch.Tensor,
    ) -> torch.Tensor:
        """Predict confidence score.

        Args:
            situation: (D,) or (B, D) situation embedding.
            context_summary: (C,) or (B, C) context summary features.

        Returns:
            Confidence score: scalar or (B,) in [0, 1].
        """
        squeeze = False
        if situation.dim() == 1:
            squeeze = True
            situation = situation.unsqueeze(0)
            context_summary = context_summary.unsqueeze(0)

        x = torch.cat([situation, context_summary], dim=-1)
        conf = self.estimator(x).squeeze(-1)

        if squeeze:
            conf = conf.squeeze(0)
        return conf

    def estimate(
        self,
        situation: torch.Tensor,
        context: ContextSummary,
    ) -> ConfidenceResult:
        """High-level confidence estimation.

        Args:
            situation: Situation embedding tensor.
            context: Context quality summary.

        Returns:
            ConfidenceResult with score, action, entropy, certainty, and fallback.
        """
        ctx_tensor = context.to_tensor().to(situation.device)
        with torch.no_grad():
            conf = self.forward(situation, ctx_tensor).item()

        return ConfidenceResult.from_confidence(conf)


# ── Calibration Loss ─────────────────────────────────────────────────

class CalibrationLoss(nn.Module):
    """Combined loss for confidence calibration.

    Components:
    1. BCE loss: predicted confidence vs actual outcome (0 or 1)
    2. Calibration penalty: binned reliability penalty (soft ECE)

    The BCE trains directional accuracy. The calibration penalty
    ensures the magnitudes are meaningful.
    """

    def __init__(self, bce_weight: float = 1.0, cal_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.cal_weight = cal_weight

    def forward(
        self,
        predicted: torch.Tensor,
        actual: torch.Tensor,
    ) -> torch.Tensor:
        """Compute calibration loss.

        Args:
            predicted: (B,) predicted confidence scores in [0, 1].
            actual: (B,) actual outcomes (0 = wrong, 1 = correct).

        Returns:
            Scalar loss.
        """
        # BCE component
        bce = F.binary_cross_entropy(predicted, actual)

        # Soft calibration component: |mean(predicted) - mean(actual)|
        # This is a differentiable approximation of ECE with 1 bin
        cal = (predicted.mean() - actual.mean()).abs()

        return self.bce_weight * bce + self.cal_weight * cal


def expected_calibration_error(
    predicted: torch.Tensor,
    actual: torch.Tensor,
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Lower is better. Perfect calibration = 0.0.
    Target: ECE < 0.1.

    Args:
        predicted: (N,) predicted confidence scores.
        actual: (N,) actual outcomes (0 or 1).
        num_bins: Number of calibration bins.

    Returns:
        ECE as float.
    """
    predicted = predicted.detach()
    actual = actual.detach()

    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total = len(predicted)

    for i in range(num_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (predicted >= lo) & (predicted < hi)
        if i == num_bins - 1:  # include 1.0 in last bin
            mask = mask | (predicted == hi)

        count = mask.sum().item()
        if count == 0:
            continue

        avg_conf = predicted[mask].mean().item()
        avg_acc = actual[mask].mean().item()
        ece += (count / total) * abs(avg_conf - avg_acc)

    return ece
