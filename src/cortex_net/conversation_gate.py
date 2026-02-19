"""Conversation Gate — learned relevance scoring for conversation history.

Same architecture as Memory Gate (bilinear scoring, identity init) but applied
to conversation turns instead of long-term memories. Decides which history turns
are relevant to the current situation, replacing a fixed sliding window.

Key insight: a new task after a long debugging discussion should see only the
new instruction, not 10 turns of prior context. But a multi-turn debugging
session should keep all recent turns. The gate learns this from feedback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ConversationContext:
    """Selected conversation turns with relevance scores."""
    messages: list[dict]        # {"role": ..., "content": ...}
    scores: list[float]         # relevance score per turn
    selected_indices: list[int] # which turns were selected


class ConversationGate(nn.Module):
    """Bilinear scorer for conversation turn relevance.

    Given a situation embedding and conversation turn embeddings,
    scores each turn's relevance to the current situation.

    Architecture mirrors MemoryGate:
    - Bilinear: score = situation^T W turn_embedding
    - Identity-initialized W: untrained gate ≈ cosine similarity
    - Learns to weight turns by relevance through feedback signal
    """

    def __init__(self, dim: int = 384, max_turns: int = 20):
        super().__init__()
        self.dim = dim
        self.max_turns = max_turns

        # Bilinear scoring matrix — identity init for cosine fallback
        self.W = nn.Parameter(torch.eye(dim))

        # Learnable threshold for inclusion (sigmoid applied)
        self.threshold_logit = nn.Parameter(torch.tensor(0.0))

        # Recency bias — learned weight for position-based decay
        self.recency_weight = nn.Parameter(torch.tensor(0.3))

        # Trained flag
        self.register_buffer("_trained", torch.tensor(False))

    @property
    def trained(self) -> bool:
        return bool(self._trained.item())

    def score_turns(
        self,
        situation: torch.Tensor,    # (dim,) current situation embedding
        turn_embeddings: torch.Tensor,  # (num_turns, dim)
    ) -> torch.Tensor:
        """Score each conversation turn's relevance to current situation.

        Returns (num_turns,) tensor of scores in [0, 1].
        """
        # Bilinear scoring: score_i = situation^T W turn_i
        # situation: (dim,) → (1, dim)
        # W: (dim, dim)
        # turn_embeddings: (num_turns, dim)
        projected = situation @ self.W  # (dim,)
        raw_scores = turn_embeddings @ projected  # (num_turns,)

        # Recency bias: more recent turns get a boost
        num_turns = turn_embeddings.shape[0]
        positions = torch.arange(num_turns, device=situation.device, dtype=torch.float32)
        recency = positions / max(num_turns - 1, 1)  # 0 (oldest) → 1 (newest)
        recency_bias = torch.sigmoid(self.recency_weight) * recency

        # Combine semantic relevance with recency
        scores = torch.sigmoid(raw_scores + recency_bias)

        return scores

    def forward(
        self,
        situation: torch.Tensor,
        turn_embeddings: torch.Tensor,
        min_turns: int = 1,
        max_select: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select relevant turns.

        Returns:
            selected_mask: (num_turns,) boolean mask
            scores: (num_turns,) relevance scores
        """
        scores = self.score_turns(situation, turn_embeddings)

        # Dynamic threshold
        threshold = torch.sigmoid(self.threshold_logit)

        # Select turns above threshold
        selected = scores >= threshold

        # Enforce min_turns: if too few selected, take top-k
        if selected.sum() < min_turns:
            k = min(min_turns, len(scores))
            _, top_idx = scores.topk(k)
            selected = torch.zeros_like(selected)
            selected[top_idx] = True

        # Enforce max_select: if too many, keep top-k
        if selected.sum() > max_select:
            # Among selected, keep top max_select
            selected_scores = scores.clone()
            selected_scores[~selected] = -1
            _, top_idx = selected_scores.topk(max_select)
            selected = torch.zeros_like(selected)
            selected[top_idx] = True

        return selected, scores

    def select_turns(
        self,
        situation: torch.Tensor,
        turn_embeddings: torch.Tensor,
        messages: list[dict],
        min_turns: int = 1,
        max_turns: int = 10,
    ) -> ConversationContext:
        """High-level API: select relevant conversation turns.

        Args:
            situation: (dim,) situation embedding
            turn_embeddings: (num_turns, dim) embeddings of each turn
            messages: list of {"role": ..., "content": ...} dicts
            min_turns: minimum turns to include
            max_turns: maximum turns to include

        Returns:
            ConversationContext with selected messages and scores
        """
        if not messages:
            return ConversationContext(messages=[], scores=[], selected_indices=[])

        with torch.no_grad():
            selected_mask, scores = self.forward(
                situation, turn_embeddings,
                min_turns=min_turns, max_select=max_turns,
            )

        selected_indices = selected_mask.nonzero(as_tuple=True)[0].tolist()
        selected_messages = [messages[i] for i in selected_indices]
        selected_scores = [scores[i].item() for i in selected_indices]

        return ConversationContext(
            messages=selected_messages,
            scores=selected_scores,
            selected_indices=selected_indices,
        )

    def training_loss(
        self,
        situation: torch.Tensor,
        turn_embeddings: torch.Tensor,
        relevance_labels: torch.Tensor,  # (num_turns,) 0 or 1
    ) -> torch.Tensor:
        """BCE loss against ground-truth relevance labels."""
        scores = self.score_turns(situation, turn_embeddings)
        return F.binary_cross_entropy(scores, relevance_labels.float())
