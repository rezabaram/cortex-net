"""Evaluation harness — metrics and baselines for Memory Gate.

Provides precision@k, recall@k, and head-to-head comparison between
the learned Memory Gate scorer and cosine similarity baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cortex_net.memory_gate import MemoryGate


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""

    precision_at_k: float
    recall_at_k: float
    k: int
    num_relevant: int


def precision_at_k(retrieved_indices: torch.Tensor | list, relevant_indices: set[int], k: int) -> float:
    """Precision@k: fraction of top-k that are relevant."""
    top_k = retrieved_indices[:k].tolist() if hasattr(retrieved_indices, 'tolist') else list(retrieved_indices[:k])
    hits = sum(1 for idx in top_k if idx in relevant_indices)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_indices: torch.Tensor | list, relevant_indices: set[int], k: int) -> float:
    """Recall@k: fraction of relevant items found in top-k."""
    if not relevant_indices:
        return 0.0
    top_k = set(retrieved_indices[:k].tolist() if hasattr(retrieved_indices, 'tolist') else retrieved_indices[:k])
    hits = len(top_k & relevant_indices)
    return hits / len(relevant_indices)


def cosine_baseline_rank(situation: torch.Tensor, memories: torch.Tensor) -> torch.Tensor:
    """Rank memories by cosine similarity (the baseline to beat)."""
    sit_norm = F.normalize(situation.unsqueeze(0), dim=-1)
    mem_norm = F.normalize(memories, dim=-1)
    scores = (sit_norm @ mem_norm.T).squeeze(0)
    return torch.argsort(scores, descending=True)


def evaluate_retrieval(
    gate: MemoryGate,
    situation: torch.Tensor,
    memories: torch.Tensor,
    relevant_indices: set[int],
    k: int = 5,
) -> tuple[RetrievalMetrics, RetrievalMetrics]:
    """Compare Memory Gate vs cosine baseline.

    Returns:
        Tuple of (gate_metrics, cosine_metrics).
    """
    # Memory Gate ranking
    gate_indices, _ = gate.select_top_k(situation, memories, k=k)

    # Cosine baseline ranking
    cosine_ranked = cosine_baseline_rank(situation, memories)

    gate_metrics = RetrievalMetrics(
        precision_at_k=precision_at_k(gate_indices, relevant_indices, k),
        recall_at_k=recall_at_k(gate_indices, relevant_indices, k),
        k=k,
        num_relevant=len(relevant_indices),
    )

    cosine_metrics = RetrievalMetrics(
        precision_at_k=precision_at_k(cosine_ranked, relevant_indices, k),
        recall_at_k=recall_at_k(cosine_ranked, relevant_indices, k),
        k=k,
        num_relevant=len(relevant_indices),
    )

    return gate_metrics, cosine_metrics


def generate_synthetic_data(
    num_memories: int = 100,
    num_relevant: int = 5,
    dim: int = 384,
    signal_strength: float = 0.3,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, set[int]]:
    """Generate synthetic evaluation data.

    Creates a situation, a pool of memories, and ground truth relevance.
    Relevant memories have a directional bias toward the situation
    (simulating topical relevance that cosine similarity partially captures).

    Args:
        num_memories: Total number of candidate memories.
        num_relevant: Number of truly relevant memories.
        dim: Embedding dimension.
        signal_strength: How much relevant memories point toward the situation.
            Higher = easier for cosine to detect. Lower = harder, more need for learned scorer.
        seed: Random seed.

    Returns:
        Tuple of (situation, memories, relevant_indices).
    """
    gen = torch.Generator().manual_seed(seed)

    situation = F.normalize(torch.randn(dim, generator=gen), dim=0)

    # Random irrelevant memories
    memories = torch.randn(num_memories, dim, generator=gen)

    # Make some memories relevant: blend with situation direction
    relevant_idx = set(range(num_relevant))
    for i in relevant_idx:
        noise = torch.randn(dim, generator=gen)
        memories[i] = signal_strength * situation + (1 - signal_strength) * noise

    # Normalize all memories
    memories = F.normalize(memories, dim=-1)

    return situation, memories, relevant_idx
