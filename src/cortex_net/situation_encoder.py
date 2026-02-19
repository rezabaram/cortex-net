"""Situation Encoder — builds a dense vector representation of the current situation.

Takes the current message, recent history, and metadata, and produces a
situation embedding that all other components consume.

The key insight: "What's the status?" means different things at 9am Monday
(standup) vs 11pm Friday (incident). Raw text embeddings lose this context.
The Situation Encoder learns to incorporate it.

Architecture: MLP that fuses text embedding + history embedding + metadata features.

Input:
    - message_embedding: (text_dim,) from sentence-transformer
    - history_embedding: (text_dim,) pooled from recent conversation
    - metadata_features: (meta_dim,) normalized numeric/categorical features

Output:
    - situation_embedding: (output_dim,) dense vector for downstream components
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SituationFeatures:
    """Raw situation features before encoding.

    This is what gets passed in from the outside world.
    The encoder handles normalization and embedding internally.
    """

    message: str = ""
    history: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Pre-computed embeddings (optional — encoder can compute them)
    message_embedding: torch.Tensor | None = None
    history_embedding: torch.Tensor | None = None


# ── Metadata feature extraction ──────────────────────────────────────

# Metadata keys we extract numeric features from
METADATA_KEYS = [
    "hour_of_day",       # 0-23, normalized to 0-1
    "day_of_week",       # 0-6 (Mon=0), normalized to 0-1
    "conversation_length",  # number of turns so far
    "message_length",    # character count of current message
    "time_since_last",   # seconds since last message
    "is_group_chat",     # 0 or 1
]

META_DIM = len(METADATA_KEYS)


def extract_metadata_features(metadata: dict[str, Any], message: str = "", history: list[str] | None = None) -> torch.Tensor:
    """Extract a fixed-size numeric feature vector from metadata.

    Missing keys default to 0. All features are normalized to roughly [0, 1].
    """
    features = []

    hour = metadata.get("hour_of_day", 12)
    features.append(hour / 24.0)

    day = metadata.get("day_of_week", 2)  # default Wednesday
    features.append(day / 6.0)

    conv_len = metadata.get("conversation_length", len(history) if history else 0)
    features.append(min(conv_len / 50.0, 1.0))  # cap at 50 turns

    msg_len = metadata.get("message_length", len(message))
    features.append(min(msg_len / 500.0, 1.0))  # cap at 500 chars

    time_since = metadata.get("time_since_last", 0)
    features.append(min(time_since / 3600.0, 1.0))  # cap at 1 hour

    is_group = float(metadata.get("is_group_chat", 0))
    features.append(is_group)

    return torch.tensor(features, dtype=torch.float32)


# ── Situation Encoder ────────────────────────────────────────────────

class SituationEncoder(nn.Module):
    """Learned situation representation.

    Fuses text embeddings with metadata to produce a situation-aware
    embedding that captures context beyond just the words.

    Architecture:
        [message_emb | history_emb | metadata] → Linear → ReLU → Linear → output

    Args:
        text_dim: Dimension of text embeddings (from sentence-transformer).
        output_dim: Dimension of output situation embedding.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout rate for regularization.
    """

    def __init__(
        self,
        text_dim: int = 384,
        output_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.output_dim = output_dim

        # Input: message_emb + history_emb + metadata
        input_dim = text_dim + text_dim + META_DIM

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Initialize final layer smaller for stable training
        nn.init.xavier_uniform_(self.encoder[-1].weight, gain=0.1)

    def forward(
        self,
        message_emb: torch.Tensor,
        history_emb: torch.Tensor,
        metadata_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a situation.

        Args:
            message_emb: (text_dim,) or (B, text_dim)
            history_emb: (text_dim,) or (B, text_dim)
            metadata_features: (META_DIM,) or (B, META_DIM)

        Returns:
            Situation embedding: (output_dim,) or (B, output_dim), L2-normalized.
        """
        # Handle unbatched input
        squeeze = False
        if message_emb.dim() == 1:
            squeeze = True
            message_emb = message_emb.unsqueeze(0)
            history_emb = history_emb.unsqueeze(0)
            metadata_features = metadata_features.unsqueeze(0)

        # Concatenate all inputs
        x = torch.cat([message_emb, history_emb, metadata_features], dim=-1)

        # Encode
        out = self.encoder(x)

        # L2 normalize for stable downstream scoring
        out = F.normalize(out, dim=-1)

        if squeeze:
            out = out.squeeze(0)

        return out

    def encode_situation(
        self,
        features: SituationFeatures,
        text_encoder=None,
    ) -> torch.Tensor:
        """High-level encoding from raw SituationFeatures.

        Args:
            features: Raw situation features.
            text_encoder: SentenceTransformer for encoding text (optional if
                          pre-computed embeddings are provided in features).

        Returns:
            Situation embedding tensor.
        """
        device = next(self.parameters()).device

        # Get message embedding
        if features.message_embedding is not None:
            msg_emb = features.message_embedding.to(device)
        elif text_encoder is not None:
            msg_emb = text_encoder.encode(
                features.message, convert_to_tensor=True
            ).to(device)
        else:
            msg_emb = torch.zeros(self.text_dim, device=device)

        # Get history embedding (mean pool over history messages)
        if features.history_embedding is not None:
            hist_emb = features.history_embedding.to(device)
        elif text_encoder is not None and features.history:
            hist_embs = text_encoder.encode(
                features.history, convert_to_tensor=True
            ).to(device)
            hist_emb = hist_embs.mean(dim=0)
        else:
            hist_emb = torch.zeros(self.text_dim, device=device)

        # Extract metadata features
        meta = extract_metadata_features(
            features.metadata, features.message, features.history
        ).to(device)

        return self.forward(msg_emb, hist_emb, meta)


# ── Contrastive training ─────────────────────────────────────────────

class SituationContrastiveLoss(nn.Module):
    """Contrastive loss for training the Situation Encoder.

    Similar situations should produce similar embeddings.
    Dissimilar situations should produce different embeddings.

    Uses NT-Xent (normalized temperature-scaled cross-entropy) loss,
    the same loss used in SimCLR.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            anchor: (D,) anchor situation embedding.
            positive: (D,) similar situation embedding.
            negatives: (N, D) dissimilar situation embeddings.

        Returns:
            Scalar loss.
        """
        # Similarities
        pos_sim = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0)) / self.temperature
        neg_sims = F.cosine_similarity(anchor.unsqueeze(0), negatives) / self.temperature

        # NT-Xent: log(exp(pos) / (exp(pos) + sum(exp(neg))))
        logits = torch.cat([pos_sim, neg_sims])
        labels = torch.zeros(1, dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits.unsqueeze(0), labels)

        return loss
