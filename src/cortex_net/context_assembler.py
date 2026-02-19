"""Context Assembler — the main pipeline.

Orchestrates all four trainable components to assemble the optimal context
window for a given situation:

    Situation → Situation Encoder → embedding
    embedding → Memory Gate → relevant memories
    embedding → Strategy Selector → strategy profile
    embedding + context → Confidence Estimator → confidence score

    Assembled Context = memories + strategy prompt + confidence framing → LLM
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from cortex_net.confidence_estimator import (
    ConfidenceEstimator,
    ConfidenceResult,
    ContextSummary,
)
from cortex_net.memory_gate import MemoryGate
from cortex_net.situation_encoder import (
    SituationEncoder,
    SituationFeatures,
    extract_metadata_features,
)
from cortex_net.state_manager import CheckpointMetadata, StateManager
from cortex_net.strategy_selector import (
    StrategyProfile,
    StrategyRegistry,
    StrategySelection,
    StrategySelector,
)


@dataclass
class AssembledContext:
    """The fully assembled context ready for the LLM."""

    # Retrieved memories
    memories: list[str]
    memory_scores: list[float]

    # Selected strategy
    strategy: StrategySelection

    # Confidence assessment
    confidence: ConfidenceResult

    # Situation embedding (for downstream use)
    situation_embedding: list[float] = field(default_factory=list)

    # Metadata
    method: str = "cortex-net"

    @property
    def prompt_prefix(self) -> str:
        """Generate a prompt prefix from the assembled context."""
        parts = []

        # Strategy framing
        if self.strategy.strategy.prompt_framing:
            parts.append(self.strategy.strategy.prompt_framing)

        # Confidence framing
        if self.confidence.action == "hedge":
            parts.append("Note: confidence is moderate. Include caveats where appropriate.")
        elif self.confidence.action == "escalate":
            parts.append("Note: confidence is low. State uncertainty clearly. Suggest alternatives or ask for clarification.")

        # Retrieved memories
        if self.memories:
            parts.append("\nRelevant context:")
            for i, (mem, score) in enumerate(zip(self.memories, self.memory_scores)):
                parts.append(f"  [{i+1}] (relevance: {score:.2f}) {mem}")

        return "\n".join(parts)


class ContextAssembler:
    """The full Context Assembly Network.

    Wires all four components together into a single inference pipeline.
    Manages state for all components.

    Args:
        text_dim: Sentence-transformer embedding dimension.
        situation_dim: Situation Encoder output dimension.
        num_strategies: Number of strategy profiles.
        state_dir: Checkpoint directory.
        embedding_model: Sentence-transformer model name.
        device: PyTorch device.
    """

    def __init__(
        self,
        text_dim: int = 384,
        num_strategies: int = 10,
        state_dir: str | Path = "./state",
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.text_dim = text_dim
        # Situation dim matches text dim — keeps cosine fallback working,
        # allows identity init for bilinear W, simplifies everything.
        self.situation_dim = text_dim

        # Text encoder (frozen)
        self.text_encoder = SentenceTransformer(embedding_model, device=device)

        # Trainable components
        self.situation_encoder = SituationEncoder(
            text_dim=text_dim, output_dim=text_dim,
        ).to(self.device)

        self.memory_gate = MemoryGate(
            situation_dim=text_dim, memory_dim=text_dim,
        ).to(self.device)

        self.strategy_registry = StrategyRegistry()

        self.strategy_selector = StrategySelector(
            situation_dim=text_dim, num_strategies=len(self.strategy_registry),
        ).to(self.device)

        self.confidence_estimator = ConfidenceEstimator(
            situation_dim=text_dim,
        ).to(self.device)
        self.state_manager = StateManager(state_dir)

    def assemble(
        self,
        query: str,
        candidate_memories: list[str],
        k: int = 5,
        history: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        explore: bool = False,
    ) -> AssembledContext:
        """Assemble the optimal context for a query.

        This is the main entry point. Runs all four components:
        1. Encode the situation
        2. Gate relevant memories
        3. Select a strategy
        4. Estimate confidence

        Args:
            query: Current message/question.
            candidate_memories: Pool of candidate memory texts.
            k: Number of memories to retrieve.
            history: Recent conversation history.
            metadata: Situation metadata.
            explore: Whether to allow strategy exploration.

        Returns:
            AssembledContext ready for the LLM.
        """
        # 1. Encode text
        query_emb = self.text_encoder.encode(query, convert_to_tensor=True).to(self.device)
        memory_embs = self.text_encoder.encode(candidate_memories, convert_to_tensor=True).to(self.device)

        if history:
            hist_embs = self.text_encoder.encode(history, convert_to_tensor=True).to(self.device)
            hist_emb = hist_embs.mean(dim=0)
        else:
            hist_emb = torch.zeros(self.text_dim, device=self.device)

        meta_features = extract_metadata_features(
            metadata or {}, query, history or []
        ).to(self.device)

        # 2. Situation Encoder
        with torch.no_grad():
            situation = self.situation_encoder(query_emb, hist_emb, meta_features)

            # 3. Memory Gate
            indices, scores = self.memory_gate.select_top_k(situation, memory_embs, k=k)

            # 4. Strategy Selector
            strategy = self.strategy_selector.select(
                situation, self.strategy_registry, explore=explore
            )

            # 5. Confidence Estimator
            indices_list = indices.cpu().tolist()
            scores_list = scores.cpu().tolist()

            ctx_summary = ContextSummary(
                num_memories_retrieved=len(indices_list),
                top_memory_score=scores_list[0] if scores_list else 0.0,
                mean_memory_score=sum(scores_list) / len(scores_list) if scores_list else 0.0,
                score_spread=(scores_list[0] - scores_list[-1]) if len(scores_list) > 1 else 0.0,
                strategy_confidence=strategy.confidence,
                query_length=len(query),
                history_length=len(history or []),
            )
            confidence = self.confidence_estimator.estimate(situation, ctx_summary)

        selected_memories = [candidate_memories[i] for i in indices_list]

        return AssembledContext(
            memories=selected_memories,
            memory_scores=scores_list,
            strategy=strategy,
            confidence=confidence,
            situation_embedding=situation.cpu().tolist(),
        )

    def save(self) -> None:
        """Save all component checkpoints."""
        components = [
            ("situation_encoder", self.situation_encoder),
            ("memory_gate", self.memory_gate),
            ("strategy_selector", self.strategy_selector),
            ("confidence_estimator", self.confidence_estimator),
        ]
        for name, module in components:
            meta = CheckpointMetadata(component_name=name)
            self.state_manager.save(module, None, meta)

    def load(self) -> bool:
        """Load all component checkpoints. Returns True if any were loaded."""
        loaded = False
        components = [
            ("situation_encoder", self.situation_encoder),
            ("memory_gate", self.memory_gate),
            ("strategy_selector", self.strategy_selector),
            ("confidence_estimator", self.confidence_estimator),
        ]
        for name, module in components:
            meta = self.state_manager.load(module, component_name=name)
            if meta is not None:
                loaded = True
        return loaded

    def parameter_count(self) -> dict[str, int]:
        """Count trainable parameters per component."""
        return {
            "situation_encoder": sum(p.numel() for p in self.situation_encoder.parameters()),
            "memory_gate": sum(p.numel() for p in self.memory_gate.parameters()),
            "strategy_selector": sum(p.numel() for p in self.strategy_selector.parameters()),
            "confidence_estimator": sum(p.numel() for p in self.confidence_estimator.parameters()),
            "total": sum(
                p.numel()
                for m in [self.situation_encoder, self.memory_gate, self.strategy_selector, self.confidence_estimator]
                for p in m.parameters()
            ),
        }
