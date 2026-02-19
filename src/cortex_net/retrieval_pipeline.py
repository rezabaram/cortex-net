"""Retrieval Pipeline — end-to-end memory retrieval with the Memory Gate.

Drop-in replacement for cosine-similarity-based retrieval. Wraps:
- Embedding generation (via sentence-transformers)
- Memory Gate scoring (learned bilinear) or cosine fallback
- State management (auto-save/load checkpoints)
- Interaction logging (for continuous training)
- Online training from logged interactions

Usage:
    pipeline = RetrievalPipeline(state_dir="./state", log_dir="./logs")
    pipeline.load()  # resume from checkpoint if available

    # Retrieve
    results = pipeline.retrieve("What's the deployment process?", candidate_texts, k=5)

    # Log outcome
    pipeline.log_outcome(results.interaction_id, signal="positive", referenced=[0, 2])

    # Train on collected data
    metrics = pipeline.train(epochs=5)

    # Save
    pipeline.save()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from cortex_net.eval import precision_at_k, recall_at_k
from cortex_net.interaction_log import (
    InteractionLogger,
    InteractionRecord,
    OutcomeData,
    OutcomeSignal,
    RetrievalData,
    SituationData,
    UsageData,
)
from cortex_net.memory_gate import MemoryGate
from cortex_net.situation_encoder import SituationEncoder, SituationFeatures, extract_metadata_features
from cortex_net.state_manager import CheckpointMetadata, StateManager


@dataclass
class RetrievalResult:
    """Result of a retrieval query."""

    indices: list[int]
    scores: list[float]
    texts: list[str]
    interaction_id: str
    method: str  # "memory_gate" or "cosine"


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""

    epochs: int = 0
    total_steps: int = 0
    avg_loss: float = 0.0
    final_loss: float = 0.0
    num_training_pairs: int = 0


@dataclass
class BenchmarkResult:
    """Head-to-head comparison results."""

    gate_precision_at_k: float = 0.0
    gate_recall_at_k: float = 0.0
    cosine_precision_at_k: float = 0.0
    cosine_recall_at_k: float = 0.0
    k: int = 5
    num_queries: int = 0
    gate_wins: int = 0
    cosine_wins: int = 0
    ties: int = 0


class RetrievalPipeline:
    """End-to-end retrieval pipeline with learned Memory Gate.

    Combines embedding, scoring, logging, and training into a single
    interface. Manages its own state on disk for full resumability.

    Args:
        state_dir: Directory for model checkpoints.
        log_dir: Directory for interaction logs.
        embedding_model: Sentence-transformers model name.
        embedding_dim: Dimension of embeddings (must match the model).
        device: PyTorch device.
        lr: Learning rate for Memory Gate training.
    """

    def __init__(
        self,
        state_dir: str | Path = "./state",
        log_dir: str | Path = "./logs",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        situation_dim: int | None = None,
        device: str = "cpu",
        lr: float = 1e-3,
    ) -> None:
        self.device = torch.device(device)
        self.embedding_dim = embedding_dim
        self.lr = lr

        # Text encoder
        self.text_encoder = SentenceTransformer(embedding_model, device=device)

        # Situation Encoder (optional — if situation_dim is set, use it)
        self.situation_dim = situation_dim
        if situation_dim is not None:
            self.situation_encoder = SituationEncoder(
                text_dim=embedding_dim,
                output_dim=situation_dim,
            ).to(self.device)
            gate_situation_dim = situation_dim
        else:
            self.situation_encoder = None
            gate_situation_dim = embedding_dim

        # Memory Gate
        self.gate = MemoryGate(
            situation_dim=gate_situation_dim,
            memory_dim=embedding_dim,
        ).to(self.device)

        # Optimizer covers both encoder and gate
        params = list(self.gate.parameters())
        if self.situation_encoder is not None:
            params += list(self.situation_encoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        # State & logging
        self.state_manager = StateManager(state_dir)
        self.logger = InteractionLogger(log_dir)

        # Track training progress
        self._step = 0
        self._epoch = 0

        # Pending interactions (not yet committed with outcome)
        self._pending: dict[str, InteractionRecord] = {}

    def load(self) -> bool:
        """Load latest checkpoint. Returns True if a checkpoint was loaded."""
        loaded = False
        meta = self.state_manager.load(
            self.gate, component_name="memory_gate"
        )
        if meta is not None:
            self._step = meta.step
            self._epoch = meta.epoch
            loaded = True

        if self.situation_encoder is not None:
            enc_meta = self.state_manager.load(
                self.situation_encoder, component_name="situation_encoder"
            )
            if enc_meta is not None:
                loaded = True

        return loaded

    def save(self) -> Path:
        """Save current state to checkpoint."""
        gate_meta = CheckpointMetadata(
            component_name="memory_gate",
            epoch=self._epoch,
            step=self._step,
        )
        path = self.state_manager.save(self.gate, None, gate_meta)

        if self.situation_encoder is not None:
            enc_meta = CheckpointMetadata(
                component_name="situation_encoder",
                epoch=self._epoch,
                step=self._step,
            )
            self.state_manager.save(self.situation_encoder, None, enc_meta)

        return path

    def embed(self, texts: str | list[str]) -> torch.Tensor:
        """Embed text(s) using the sentence transformer."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.text_encoder.encode(texts, convert_to_tensor=True)
        return embeddings.to(self.device)

    def encode_situation(
        self,
        query: str,
        history: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Encode a query into a situation embedding.

        If Situation Encoder is enabled, fuses text + history + metadata.
        Otherwise, returns the raw text embedding.
        """
        query_emb = self.embed(query).squeeze(0)

        if self.situation_encoder is None:
            return query_emb

        # Compute history embedding
        if history:
            hist_embs = self.embed(history)
            hist_emb = hist_embs.mean(dim=0)
        else:
            hist_emb = torch.zeros(self.embedding_dim, device=self.device)

        meta = extract_metadata_features(
            metadata or {}, query, history or []
        ).to(self.device)

        return self.situation_encoder(query_emb, hist_emb, meta)

    def retrieve(
        self,
        query: str,
        candidate_texts: list[str],
        k: int = 5,
        history: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Retrieve the top-k most relevant memories for a query.

        Args:
            query: The current situation/question.
            candidate_texts: Pool of candidate memory texts.
            k: Number of memories to retrieve.
            history: Recent conversation history.
            metadata: Situation metadata (channel, user, etc.).

        Returns:
            RetrievalResult with indices, scores, and texts.
        """
        # Encode situation (uses Situation Encoder if available, else raw text)
        situation_emb = self.encode_situation(query, history, metadata)
        candidate_embs = self.embed(candidate_texts)

        # Score with Memory Gate (falls back to cosine if untrained)
        with torch.no_grad():
            indices, scores = self.gate.select_top_k(situation_emb, candidate_embs, k=k)

        indices_list = indices.cpu().tolist()
        scores_list = scores.cpu().tolist()
        selected_texts = [candidate_texts[i] for i in indices_list]

        method = "situation_encoder+memory_gate" if self.situation_encoder is not None else (
            "memory_gate" if self.gate.trained else "cosine"
        )

        # Log the interaction (pending outcome)
        situation = SituationData(
            message=query,
            history=history or [],
            metadata=metadata or {},
            embedding=situation_emb.detach().cpu().tolist(),
        )
        record = self.logger.begin(situation)
        record.retrieval = RetrievalData(
            candidate_ids=[str(i) for i in range(len(candidate_texts))],
            scores=[float(s) for s in scores_list],
            selected_indices=indices_list,
            method=method,
        )
        self._pending[record.id] = record

        return RetrievalResult(
            indices=indices_list,
            scores=scores_list,
            texts=selected_texts,
            interaction_id=record.id,
            method=method,
        )

    def log_outcome(
        self,
        interaction_id: str,
        signal: str = OutcomeSignal.UNKNOWN,
        source: str = "user_feedback",
        referenced: list[int] | None = None,
        score: float | None = None,
        details: str = "",
    ) -> None:
        """Record the outcome for a pending interaction.

        Args:
            interaction_id: The interaction_id from RetrievalResult.
            signal: "positive", "negative", "neutral", or "unknown".
            source: Where the signal came from.
            referenced: Which retrieved indices were actually used/referenced.
            score: Optional graded score 0.0-1.0.
            details: Free text details.
        """
        record = self._pending.pop(interaction_id, None)
        if record is None:
            return

        record.usage = UsageData(
            referenced_indices=referenced or [],
            used_in_response=bool(referenced),
        )
        record.outcome = OutcomeData(
            signal=signal,
            source=source,
            score=score,
            details=details,
        )
        self.logger.commit(record)

    def train(self, epochs: int = 1, margin: float = 1.0) -> TrainingMetrics:
        """Train the Memory Gate on logged interactions.

        Extracts training pairs from interaction logs and runs
        contrastive training. Call this periodically or after collecting
        enough labeled interactions.

        Args:
            epochs: Number of passes over the training data.
            margin: Contrastive loss margin.

        Returns:
            Training metrics.
        """
        pairs = self.logger.extract_training_pairs()
        if not pairs:
            return TrainingMetrics(num_training_pairs=0)

        total_loss = 0.0
        steps = 0

        for epoch in range(epochs):
            for situation_data, pos_indices, neg_indices in pairs:
                # We need embeddings. If stored, use them; otherwise skip
                if situation_data.embedding is None:
                    continue

                sit_emb = torch.tensor(situation_data.embedding, device=self.device)

                # For training, we need memory embeddings too.
                # In a real system these would be stored or re-embedded.
                # For now, we use the logged data structure.
                # This is a limitation — Phase 2 will address it with
                # a proper embedding cache.

                # Skip if we don't have enough data
                if not pos_indices or not neg_indices:
                    continue

                # Generate synthetic positive/negative embeddings from indices
                # In production, these would come from the embedding store
                # For now, create directional embeddings for training signal
                pos_embs = self._make_training_embeddings(sit_emb, pos_indices, positive=True)
                neg_embs = self._make_training_embeddings(sit_emb, neg_indices, positive=False)

                loss = self.gate.train_step(
                    self.optimizer, sit_emb, pos_embs, neg_embs, margin=margin
                )
                total_loss += loss
                steps += 1
                self._step += 1

            self._epoch += 1

        avg_loss = total_loss / steps if steps > 0 else 0.0
        return TrainingMetrics(
            epochs=epochs,
            total_steps=steps,
            avg_loss=avg_loss,
            final_loss=total_loss / max(steps, 1),
            num_training_pairs=len(pairs),
        )

    def benchmark(
        self,
        queries: list[str],
        candidate_texts: list[str],
        ground_truth: list[set[int]],
        k: int = 5,
    ) -> BenchmarkResult:
        """Run head-to-head benchmark: Memory Gate vs cosine similarity.

        Args:
            queries: List of query texts.
            candidate_texts: Pool of candidate memories (same for all queries).
            ground_truth: List of sets of relevant candidate indices per query.
            k: Number of results to compare.

        Returns:
            BenchmarkResult with aggregate metrics.
        """
        candidate_embs = self.embed(candidate_texts)

        gate_precisions = []
        gate_recalls = []
        cosine_precisions = []
        cosine_recalls = []
        gate_wins = 0
        cosine_wins = 0
        ties = 0

        for query, relevant in zip(queries, ground_truth):
            query_emb = self.embed(query).squeeze(0)

            # Memory Gate ranking
            with torch.no_grad():
                gate_idx, _ = self.gate.select_top_k(query_emb, candidate_embs, k=k)

            # Cosine ranking
            cosine_scores = F.normalize(query_emb.unsqueeze(0), dim=-1) @ F.normalize(
                candidate_embs, dim=-1
            ).T
            cosine_idx = torch.argsort(cosine_scores.squeeze(0), descending=True)[:k]

            gp = precision_at_k(gate_idx, relevant, k)
            gr = recall_at_k(gate_idx, relevant, k)
            cp = precision_at_k(cosine_idx, relevant, k)
            cr = recall_at_k(cosine_idx, relevant, k)

            gate_precisions.append(gp)
            gate_recalls.append(gr)
            cosine_precisions.append(cp)
            cosine_recalls.append(cr)

            if gp > cp:
                gate_wins += 1
            elif cp > gp:
                cosine_wins += 1
            else:
                ties += 1

        n = len(queries)
        return BenchmarkResult(
            gate_precision_at_k=sum(gate_precisions) / n if n else 0,
            gate_recall_at_k=sum(gate_recalls) / n if n else 0,
            cosine_precision_at_k=sum(cosine_precisions) / n if n else 0,
            cosine_recall_at_k=sum(cosine_recalls) / n if n else 0,
            k=k,
            num_queries=n,
            gate_wins=gate_wins,
            cosine_wins=cosine_wins,
            ties=ties,
        )

    def _make_training_embeddings(
        self,
        situation: torch.Tensor,
        indices: list[int],
        positive: bool,
    ) -> torch.Tensor:
        """Create training embeddings from indices.

        Temporary approach: generates directional embeddings.
        Positive embeddings point toward the situation (simulating relevance).
        Negative embeddings are random directions.

        TODO: Replace with actual memory embeddings from an embedding cache
        once the pipeline is integrated with a real memory store.
        """
        n = len(indices)
        dim = situation.shape[0]
        gen = torch.Generator(device=self.device)
        # Use index as seed component for reproducibility
        gen.manual_seed(sum(indices) + (1 if positive else 0))

        noise = torch.randn(n, dim, generator=gen, device=self.device)
        if positive:
            # Blend toward situation
            embs = 0.3 * situation.unsqueeze(0) + 0.7 * noise
        else:
            embs = noise

        return F.normalize(embs, dim=-1)
