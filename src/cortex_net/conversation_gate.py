"""Conversation Gate — learned relevance scoring for conversation history.

Two-tier architecture:
1. Bilinear scorer (pointwise): scores each turn independently against situation
2. Cross-attention layer (contextual): refines scores by letting turns attend to
   each other and to the situation. Captures Q&A pairs, reasoning chains, and
   compositional relevance.

The attention layer is initialized to pass through the bilinear scores unchanged,
so untrained behavior = bilinear gate. Training can only improve, never regress.
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


class BilinearScorer(nn.Module):
    """Pointwise bilinear relevance scorer (original gate mechanism)."""

    def __init__(self, dim: int = 384):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim))
        self.recency_weight = nn.Parameter(torch.tensor(0.3))
        self.decay_rate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        situation: torch.Tensor,     # (dim,)
        turn_embeddings: torch.Tensor,  # (num_turns, dim)
    ) -> torch.Tensor:
        """Returns (num_turns,) raw scores (pre-sigmoid)."""
        projected = situation @ self.W  # (dim,)
        raw_scores = turn_embeddings @ projected  # (num_turns,)

        num_turns = turn_embeddings.shape[0]
        positions = torch.arange(num_turns, device=situation.device, dtype=torch.float32)
        recency = positions / max(num_turns - 1, 1)
        recency_bias = torch.sigmoid(self.recency_weight) * recency

        age = 1.0 - recency
        decay = torch.sigmoid(self.decay_rate)
        temporal_penalty = decay * age

        return raw_scores + recency_bias - temporal_penalty


class ContextualAttention(nn.Module):
    """Cross-attention that refines pointwise scores using inter-turn context.

    Architecture:
    - Each turn is represented as [turn_embedding; bilinear_score]
    - Single-head cross-attention: situation queries, turns are keys/values
    - Self-attention among turns to capture Q&A pairs and reasoning chains
    - Output: refined relevance score per turn

    Identity initialization: attention outputs zero initially, so
    final_score = bilinear_score + 0 = bilinear_score (untrained = passthrough).
    """

    def __init__(self, dim: int = 384, head_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.head_dim = head_dim

        # Project turn embeddings + bilinear score to attention space
        self.turn_proj = nn.Linear(dim + 1, head_dim)

        # Situation query projection
        self.situation_proj = nn.Linear(dim, head_dim)

        # Self-attention among turns (captures inter-turn relationships)
        self.self_attn_q = nn.Linear(head_dim, head_dim)
        self.self_attn_k = nn.Linear(head_dim, head_dim)
        self.self_attn_v = nn.Linear(head_dim, head_dim)

        # Cross-attention: situation queries turns
        self.cross_q = nn.Linear(head_dim, head_dim)
        self.cross_k = nn.Linear(head_dim, head_dim)
        self.cross_v = nn.Linear(head_dim, head_dim)

        # Output: refined score per turn
        self.score_head = nn.Linear(head_dim, 1)

        # Residual gate: learned blend between bilinear and attention scores
        # Initialized to 0 → sigmoid(0) = 0.5, but we want full passthrough initially
        self.residual_gate = nn.Parameter(torch.tensor(-5.0))  # sigmoid(-5) ≈ 0.007

        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** 0.5

        # Initialize score_head to near-zero so attention contribution starts at ~0
        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    def forward(
        self,
        situation: torch.Tensor,        # (dim,)
        turn_embeddings: torch.Tensor,  # (N, dim)
        bilinear_scores: torch.Tensor,  # (N,) pre-sigmoid
    ) -> torch.Tensor:
        """Returns refined raw scores (pre-sigmoid), shape (N,)."""
        N = turn_embeddings.shape[0]
        if N == 0:
            return bilinear_scores

        # Step 1: Build turn representations [embedding; bilinear_score]
        score_feature = bilinear_scores.unsqueeze(-1)  # (N, 1)
        turn_features = torch.cat([turn_embeddings, score_feature], dim=-1)  # (N, dim+1)
        turn_hidden = self.turn_proj(turn_features)  # (N, head_dim)

        # Step 2: Self-attention among turns
        # This lets the model learn "if turn i is selected, turn i+1 should be too"
        sa_q = self.self_attn_q(turn_hidden)  # (N, head_dim)
        sa_k = self.self_attn_k(turn_hidden)
        sa_v = self.self_attn_v(turn_hidden)

        sa_weights = (sa_q @ sa_k.T) / self.scale  # (N, N)
        sa_weights = F.softmax(sa_weights, dim=-1)
        sa_weights = self.dropout(sa_weights)
        turn_context = sa_weights @ sa_v  # (N, head_dim)

        # Residual connection
        turn_hidden = turn_hidden + turn_context

        # Step 3: Cross-attention — situation queries the contextualized turns
        sit_hidden = self.situation_proj(situation).unsqueeze(0)  # (1, head_dim)
        ca_q = self.cross_q(turn_hidden)   # (N, head_dim) — each turn as query
        ca_k = self.cross_k(sit_hidden)    # (1, head_dim) — situation as key
        ca_v = self.cross_v(sit_hidden)    # (1, head_dim) — situation as value

        # Each turn attends to the situation (how relevant is this turn to situation?)
        ca_weights = (ca_q @ ca_k.T) / self.scale  # (N, 1)
        ca_weights = torch.sigmoid(ca_weights)  # sigmoid for per-turn relevance
        situation_context = ca_weights * ca_v  # (N, head_dim)

        # Combine self-attention context with cross-attention context
        combined = turn_hidden + situation_context  # (N, head_dim)

        # Step 4: Score head
        attention_scores = self.score_head(combined).squeeze(-1)  # (N,)

        # Step 5: Residual blend — gate controls bilinear vs attention contribution
        gate = torch.sigmoid(self.residual_gate)
        refined_scores = (1 - gate) * bilinear_scores + gate * attention_scores

        return refined_scores


class ConversationGate(nn.Module):
    """Two-tier conversation context selector.

    Tier 1: Bilinear pointwise scoring (fast, works from day 1)
    Tier 2: Cross-attention contextual refinement (learns inter-turn relationships)

    Untrained: residual_gate ≈ 0, so output = bilinear scores (zero regression risk)
    Trained: gate opens, attention refines scores (captures Q&A pairs, chains)
    """

    def __init__(self, dim: int = 384, head_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        # Tier 1: Pointwise bilinear scorer
        self.bilinear = BilinearScorer(dim)

        # Tier 2: Contextual attention refinement
        self.attention = ContextualAttention(dim, head_dim, dropout)

        # Learnable threshold for inclusion (sigmoid applied)
        self.threshold_logit = nn.Parameter(torch.tensor(0.0))

        # Trained flag
        self.register_buffer("_trained", torch.tensor(False))

    @property
    def trained(self) -> bool:
        return bool(self._trained.item())

    # Backward compat: expose bilinear params at top level for existing checkpoints
    @property
    def W(self):
        return self.bilinear.W

    @property
    def recency_weight(self):
        return self.bilinear.recency_weight

    @property
    def decay_rate(self):
        return self.bilinear.decay_rate

    def score_turns(
        self,
        situation: torch.Tensor,
        turn_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Score each turn's relevance. Returns (num_turns,) in [0, 1]."""
        if turn_embeddings.shape[0] == 0:
            return torch.tensor([], device=situation.device)

        # Tier 1: Bilinear pointwise scores
        bilinear_raw = self.bilinear(situation, turn_embeddings)

        # Tier 2: Attention refinement (if >1 turn — no self-attention for single turn)
        if turn_embeddings.shape[0] > 1:
            refined_raw = self.attention(situation, turn_embeddings, bilinear_raw)
        else:
            refined_raw = bilinear_raw

        return torch.sigmoid(refined_raw)

    def forward(
        self,
        situation: torch.Tensor,
        turn_embeddings: torch.Tensor,
        min_turns: int = 0,
        max_select: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select relevant turns.

        Returns:
            selected_mask: (num_turns,) boolean mask
            scores: (num_turns,) relevance scores
        """
        scores = self.score_turns(situation, turn_embeddings)

        if len(scores) == 0:
            return torch.tensor([], dtype=torch.bool), scores

        # Adaptive threshold
        abs_threshold = torch.sigmoid(self.threshold_logit)
        if len(scores) > 4:
            rel_threshold = scores.mean() + 0.5 * scores.std()
            threshold = torch.max(abs_threshold, rel_threshold)
        else:
            threshold = abs_threshold

        selected = scores >= threshold

        # Enforce min_turns
        if min_turns > 0 and selected.sum() < min_turns:
            k = min(min_turns, len(scores))
            _, top_idx = scores.topk(k)
            selected = torch.zeros_like(selected)
            selected[top_idx] = True

        # Enforce max_select
        if selected.sum() > max_select:
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
        min_turns: int = 0,
        max_turns: int = 10,
    ) -> ConversationContext:
        """High-level API: select relevant conversation turns."""
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

    @property
    def condition_number(self) -> float:
        """Condition number of the bilinear weight matrix.
        
        High condition number → ill-conditioned → numerical instability.
        Returns inf if W has zero singular values.
        """
        W = self.bilinear.W.detach()
        # Use SVD for stable condition number computation
        try:
            _, s, _ = torch.svd(W)
            # Condition number = max_singular / min_singular
            min_s = s.min().clamp(min=1e-10)
            max_s = s.max()
            return (max_s / min_s).item()
        except RuntimeError:
            return float('inf')

    @property
    def stability_score(self) -> float:
        """Stability score in [0, 1], where 1 = perfectly stable.
        
        Computed as inverse of condition number, clamped.
        Score > 0.5 is considered stable.
        """
        cond = self.condition_number
        if cond == float('inf'):
            return 0.0
        # Inverse, clamped: 1/cond=1.0 (perfect), 1/100=0.01 (poor)
        score = 1.0 / cond
        return float(torch.clamp(torch.tensor(score), 0.0, 1.0).item())

    def is_stable(self, threshold: float = 0.5) -> bool:
        """Check if the gate is numerically stable.
        
        Args:
            threshold: Minimum stability_score required (default 0.5)
            
        Returns:
            True if stability_score >= threshold
        """
        return self.stability_score >= threshold

    def diagnose(self) -> dict:
        """Return diagnostic information about the gate's health."""
        return {
            "is_stable": self.is_stable(),
            "stability_score": self.stability_score,
            "condition_number": self.condition_number,
            "trained": self.trained,
        }

    def training_loss(
        self,
        situation: torch.Tensor,
        turn_embeddings: torch.Tensor,
        relevance_labels: torch.Tensor,
    ) -> torch.Tensor:
        """BCE loss against ground-truth relevance labels."""
        scores = self.score_turns(situation, turn_embeddings)
        return F.binary_cross_entropy(scores, relevance_labels.float())

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load with backward compatibility for old flat checkpoint format."""
        # Check if this is an old-format checkpoint (flat W, recency_weight, decay_rate)
        if "W" in state_dict and "bilinear.W" not in state_dict:
            # Remap old flat keys to new nested structure
            new_state = {}
            remap = {
                "W": "bilinear.W",
                "recency_weight": "bilinear.recency_weight",
                "decay_rate": "bilinear.decay_rate",
            }
            for k, v in state_dict.items():
                new_state[remap.get(k, k)] = v

            # Load bilinear weights, skip missing attention weights (they stay at init)
            return super().load_state_dict(new_state, strict=False, assign=assign)
        else:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
