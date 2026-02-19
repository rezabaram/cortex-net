"""Memory Gate — learned relevance scoring for memory retrieval.

Replaces naive cosine similarity with a trained bilinear scorer that learns
which memories are actually relevant given the current situation.

Architecture: bilinear scoring function
    relevance(situation, memory) = situation · W · memory

Training signal: did retrieving this memory lead to a better outcome?
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryGate(nn.Module):
    """Bilinear scorer for memory relevance.

    Learns a weight matrix W such that:
        score(situation, memory) = situation @ W @ memory.T

    Falls back to cosine similarity when untrained (W ≈ I).

    Numerical stability:
        - Scores are clamped to prevent overflow/underflow in softmax
        - Weight matrix has optional L2 regularization
        - Input validation checks for NaN/Inf

    Args:
        situation_dim: Dimension of situation embeddings.
        memory_dim: Dimension of memory embeddings.
        init_as_identity: If True, initialize W close to identity/cosine behavior.
    """

    # Clamp bounds for numerical stability
    SCORE_CLAMP_MIN: float = -100.0
    SCORE_CLAMP_MAX: float = 100.0

    def __init__(
        self,
        situation_dim: int = 384,
        memory_dim: int = 384,
        init_as_identity: bool = True,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.situation_dim = situation_dim
        self.memory_dim = memory_dim

        # Bilinear weight matrix: situation_dim x memory_dim
        self.W = nn.Parameter(torch.empty(situation_dim, memory_dim))

        if init_as_identity and situation_dim == memory_dim:
            # Start near identity → behaves like cosine similarity initially
            nn.init.eye_(self.W)
            self.W.data += torch.randn_like(self.W) * 0.01
        else:
            nn.init.xavier_uniform_(self.W)

        # Learnable temperature: higher = softer distribution, lower = sharper
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

        # Persistent flag: included in state_dict via register_buffer
        self.register_buffer("_trained_flag", torch.tensor(0, dtype=torch.bool))

    @property
    def trained(self) -> bool:
        return bool(self._trained_flag.item())

    def score_memories(
        self,
        situation: torch.Tensor,
        memories: torch.Tensor,
        use_cosine_fallback: bool = True,
    ) -> torch.Tensor:
        """Score each memory's relevance to the situation.

        Args:
            situation: (D_s,) or (B, D_s) situation embedding.
            memories: (N, D_m) candidate memory embeddings.
            use_cosine_fallback: If True and model is untrained, use cosine similarity.

        Returns:
            Relevance scores: (N,) or (B, N).
        """
        # Input validation: check for NaN/Inf
        if torch.isnan(situation).any() or torch.isinf(situation).any():
            raise ValueError("situation contains NaN or Inf values")
        if torch.isnan(memories).any() or torch.isinf(memories).any():
            raise ValueError("memories contains NaN or Inf values")

        # Use cosine fallback if model is untrained OR weights became unstable
        use_fallback = use_cosine_fallback and (
            not self.trained or not self.is_stable()
        )
        
        if use_fallback and self.situation_dim == self.memory_dim:
            return self._cosine_scores(situation, memories)

        # Bilinear: situation @ W @ memories.T
        if situation.dim() == 1:
            # (D_s,) @ (D_s, D_m) -> (D_m,)
            projected = situation @ self.W
            # (D_m,) @ (N, D_m).T -> (N,)
            scores = projected @ memories.T
        else:
            # (B, D_s) @ (D_s, D_m) -> (B, D_m)
            projected = situation @ self.W
            # (B, D_m) @ (D_m, N) -> (B, N)
            scores = projected @ memories.T

        # Temperature scaling: sharper with low temp, softer with high temp
        scores = scores / self.temperature.clamp(min=0.01)

        # Numerical stability: clamp scores to prevent overflow/underflow downstream
        scores = torch.clamp(scores, self.SCORE_CLAMP_MIN, self.SCORE_CLAMP_MAX)

        # Check for NaN/Inf in output
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            raise RuntimeError("score_memories produced NaN or Inf - possible weight instability")

        return scores

    def select_top_k(
        self,
        situation: torch.Tensor,
        memories: torch.Tensor,
        k: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the top-k most relevant memories.

        Args:
            situation: (D_s,) situation embedding.
            memories: (N, D_m) candidate memory embeddings.
            k: Number of memories to select.

        Returns:
            Tuple of (indices, scores) for the top-k memories.
        """
        scores = self.score_memories(situation, memories)
        k = min(k, scores.shape[-1])
        top_scores, top_indices = torch.topk(scores, k, dim=-1)
        return top_indices, top_scores

    def contrastive_loss(
        self,
        situation: torch.Tensor,
        positive_memories: torch.Tensor,
        negative_memories: torch.Tensor,
        margin: float = 1.0,
        weight_decay: float = 0.0,
    ) -> torch.Tensor:
        """Contrastive loss: positives should score higher than negatives by margin.

        Args:
            situation: (D_s,) situation embedding.
            positive_memories: (P, D_m) relevant memory embeddings.
            negative_memories: (N, D_m) irrelevant memory embeddings.
            margin: Minimum score gap between positives and negatives.
            weight_decay: L2 regularization strength for weight matrix W.

        Returns:
            Scalar loss.
        """
        pos_scores = self.score_memories(situation, positive_memories, use_cosine_fallback=False)
        neg_scores = self.score_memories(situation, negative_memories, use_cosine_fallback=False)

        # Mean positive score should exceed mean negative score by margin
        loss = F.relu(margin - pos_scores.mean() + neg_scores.mean())

        # Optional weight decay for numerical stability
        if weight_decay > 0:
            loss = loss + weight_decay * torch.sum(self.W ** 2)

        return loss

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        situation: torch.Tensor,
        positive_memories: torch.Tensor,
        negative_memories: torch.Tensor,
        margin: float = 1.0,
        weight_decay: float = 0.0,
    ) -> float:
        """Single training step.

        Args:
            optimizer: Optimizer for this module's parameters.
            situation: (D_s,) situation embedding.
            positive_memories: (P, D_m) relevant memory embeddings.
            negative_memories: (N, D_m) irrelevant memory embeddings.
            margin: Contrastive margin.
            weight_decay: L2 regularization strength.

        Returns:
            Loss value as float.
        """
        optimizer.zero_grad()
        loss = self.contrastive_loss(situation, positive_memories, negative_memories, margin, weight_decay)
        loss.backward()
        optimizer.step()

        self._trained_flag.fill_(1)
        return loss.item()

    def is_stable(self, eps: float = 1e6) -> bool:
        """Check if weight matrix is numerically stable.

        Args:
            eps: Maximum allowed singular value before considering unstable.

        Returns:
            True if weights appear stable, False if they may have exploded.
        """
        # Compute spectral norm (largest singular value)
        singular_values = torch.linalg.svdvals(self.W)
        max_sv = singular_values[0].item()
        return max_sv < eps

    @staticmethod
    def _cosine_scores(situation: torch.Tensor, memories: torch.Tensor) -> torch.Tensor:
        """Cosine similarity fallback."""
        if situation.dim() == 1:
            situation = situation.unsqueeze(0)
        # Normalize
        situation_norm = F.normalize(situation, dim=-1)
        memories_norm = F.normalize(memories, dim=-1)
        scores = situation_norm @ memories_norm.T
        return scores.squeeze(0) if scores.shape[0] == 1 else scores
