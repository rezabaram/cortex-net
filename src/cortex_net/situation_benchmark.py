"""Situation Encoder benchmark — prove metadata-aware embeddings beat raw text.

Creates scenarios where the same text query needs different memories depending
on context (time, conversation history, metadata). Raw text embeddings can't
distinguish these. The Situation Encoder should.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from cortex_net.embedding_store import EmbeddingStore
from cortex_net.eval import precision_at_k
from cortex_net.memory_gate import MemoryGate
from cortex_net.situation_encoder import (
    SituationEncoder,
    SituationContrastiveLoss,
    SituationFeatures,
    extract_metadata_features,
)


@dataclass
class ContextualScenario:
    """A scenario where context changes which memories are relevant."""

    name: str
    query: str  # Same query text for both contexts
    memories: list[str]
    # Context A: e.g., morning standup
    context_a_metadata: dict
    context_a_relevant: set[int]
    # Context B: e.g., late-night incident
    context_b_metadata: dict
    context_b_relevant: set[int]
    description: str = ""


@dataclass
class ContextualResult:
    name: str
    # With Situation Encoder
    encoder_a_precision: float
    encoder_b_precision: float
    # Without (raw text only — same for both contexts)
    raw_a_precision: float
    raw_b_precision: float
    k: int

    @property
    def encoder_avg(self) -> float:
        return (self.encoder_a_precision + self.encoder_b_precision) / 2

    @property
    def raw_avg(self) -> float:
        return (self.raw_a_precision + self.raw_b_precision) / 2

    @property
    def encoder_wins(self) -> bool:
        return self.encoder_avg > self.raw_avg


def build_contextual_scenarios() -> list[ContextualScenario]:
    return [
        ContextualScenario(
            name="status_check",
            description="'What's the status?' means different things at standup vs during incident",
            query="What's the status?",
            memories=[
                # Standup-relevant (morning, routine)
                "Sprint velocity is at 23 points, on track for the release.",
                "PR #142 is waiting for review, blocked on the API team.",
                # Incident-relevant (late night, urgent)
                "The payment service is returning 503 errors since 10:45 PM.",
                "On-call engineer has been paged, investigating database connection pool exhaustion.",
                # Noise
                "The office plants need watering on Thursdays.",
                "Team lunch is at the Italian place this Friday.",
            ],
            context_a_metadata={"hour_of_day": 9, "day_of_week": 1, "is_group_chat": 1},
            context_a_relevant={0, 1},  # sprint/PR info
            context_b_metadata={"hour_of_day": 23, "day_of_week": 4, "is_group_chat": 1},
            context_b_relevant={2, 3},  # incident info
        ),
        ContextualScenario(
            name="help_request",
            description="'Can you help me?' from new vs experienced user",
            query="Can you help me with this?",
            memories=[
                # New user relevant
                "Here's our getting started guide with step-by-step setup instructions.",
                "Common first-time setup issues: make sure Docker is running and ports 3000/5432 are free.",
                # Experienced user relevant
                "Advanced debugging: attach to the running container with `docker exec -it` and check /var/log/app.",
                "For performance profiling, use the built-in flame graph at /debug/profile.",
                # Noise
                "The cafeteria has a new menu this week.",
                "Parking lot B is closed for maintenance until March.",
            ],
            context_a_metadata={"hour_of_day": 10, "conversation_length": 1, "message_length": 25},
            context_a_relevant={0, 1},  # new user, short history
            context_b_metadata={"hour_of_day": 15, "conversation_length": 50, "message_length": 25},
            context_b_relevant={2, 3},  # experienced user, long history
        ),
        ContextualScenario(
            name="quick_vs_deep",
            description="Short DM vs long group thread needs different response depth",
            query="How does the caching layer work?",
            memories=[
                # Quick answer relevant (DM, short)
                "The cache uses Redis with a 15-minute TTL. Invalidation is event-driven via pub/sub.",
                "Cache hit rate is currently 94%. Dashboard at /metrics/cache.",
                # Deep dive relevant (group, long thread)
                "Cache architecture: L1 is in-process LRU (1000 entries), L2 is Redis cluster (3 nodes). Write-through on mutations, lazy invalidation on reads.",
                "Cache consistency model: eventual consistency with 50ms propagation. Strong consistency available via cache-control: no-cache header.",
                # Noise
                "New employee onboarding is every Monday.",
                "The API documentation is at docs.internal.com.",
            ],
            context_a_metadata={"hour_of_day": 14, "is_group_chat": 0, "conversation_length": 2},
            context_a_relevant={0, 1},  # DM, quick
            context_b_metadata={"hour_of_day": 14, "is_group_chat": 1, "conversation_length": 20},
            context_b_relevant={2, 3},  # group thread, deep
        ),
    ]


def run_contextual_benchmark(
    encoder: SituationEncoder,
    gate_with_encoder: MemoryGate,
    gate_raw: MemoryGate,
    store: EmbeddingStore,
    scenarios: list[ContextualScenario] | None = None,
    k: int = 2,
) -> list[ContextualResult]:
    """Benchmark: Situation Encoder + Gate vs raw text + Gate.

    Both gates are trained. The question is whether the encoder's
    metadata-aware embeddings improve retrieval when context matters.
    """
    if scenarios is None:
        scenarios = build_contextual_scenarios()

    results = []
    for s in scenarios:
        query_text_emb = store.encode(s.query)
        memory_embs = store.encode_batch(s.memories)

        # --- With Situation Encoder ---
        meta_a = extract_metadata_features(s.context_a_metadata).to(query_text_emb.device)
        meta_b = extract_metadata_features(s.context_b_metadata).to(query_text_emb.device)
        hist_emb = torch.zeros_like(query_text_emb)  # no history text for simplicity

        with torch.no_grad():
            sit_a = encoder(query_text_emb, hist_emb, meta_a)
            sit_b = encoder(query_text_emb, hist_emb, meta_b)

            # Need to project memory embeddings to encoder output dim if different
            # For this benchmark, gate_with_encoder's situation_dim == encoder.output_dim
            enc_idx_a, _ = gate_with_encoder.select_top_k(sit_a, memory_embs, k=k)
            enc_idx_b, _ = gate_with_encoder.select_top_k(sit_b, memory_embs, k=k)

        enc_a_p = precision_at_k(enc_idx_a, s.context_a_relevant, k)
        enc_b_p = precision_at_k(enc_idx_b, s.context_b_relevant, k)

        # --- Without encoder (raw text, same for both contexts) ---
        with torch.no_grad():
            raw_idx, _ = gate_raw.select_top_k(query_text_emb, memory_embs, k=k)

        raw_a_p = precision_at_k(raw_idx, s.context_a_relevant, k)
        raw_b_p = precision_at_k(raw_idx, s.context_b_relevant, k)

        results.append(ContextualResult(
            name=s.name,
            encoder_a_precision=enc_a_p,
            encoder_b_precision=enc_b_p,
            raw_a_precision=raw_a_p,
            raw_b_precision=raw_b_p,
            k=k,
        ))

    return results


def train_contextual(
    encoder: SituationEncoder,
    gate: MemoryGate,
    store: EmbeddingStore,
    scenarios: list[ContextualScenario] | None = None,
    epochs: int = 200,
    lr: float = 1e-3,
) -> list[float]:
    """Train Situation Encoder + Memory Gate jointly on contextual scenarios.

    For each scenario, both contexts are used as training signal:
    the encoder should produce different situation embeddings that lead
    the gate to select the correct memories for each context.
    """
    if scenarios is None:
        scenarios = build_contextual_scenarios()

    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    gate_optimizer = torch.optim.Adam(gate.parameters(), lr=lr)

    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for s in scenarios:
            query_emb = store.encode(s.query)
            memory_embs = store.encode_batch(s.memories)
            hist_emb = torch.zeros_like(query_emb)

            for ctx_meta, relevant in [
                (s.context_a_metadata, s.context_a_relevant),
                (s.context_b_metadata, s.context_b_relevant),
            ]:
                meta = extract_metadata_features(ctx_meta).to(query_emb.device)
                sit = encoder(query_emb, hist_emb, meta)

                pos_idx = list(relevant)
                neg_idx = [i for i in range(len(s.memories)) if i not in relevant]
                pos = memory_embs[pos_idx]
                neg = memory_embs[neg_idx]

                loss = gate.contrastive_loss(sit, pos, neg, margin=1.0)

                enc_optimizer.zero_grad()
                gate_optimizer.zero_grad()
                loss.backward()
                enc_optimizer.step()
                gate_optimizer.step()

                total_loss += loss.item()

        gate._trained_flag.fill_(1)
        epoch_losses.append(total_loss / (len(scenarios) * 2))

    return epoch_losses


def print_contextual_report(results: list[ContextualResult]) -> str:
    lines = ["Contextual Benchmark (same query, different context)", ""]
    enc_total = 0.0
    raw_total = 0.0
    enc_wins = 0

    for r in results:
        winner = "ENC" if r.encoder_wins else ("TIE" if r.encoder_avg == r.raw_avg else "RAW")
        lines.append(
            f"  [{winner}] {r.name}: encoder={r.encoder_avg:.2f} raw={r.raw_avg:.2f}"
        )
        lines.append(f"         ctx_a: enc={r.encoder_a_precision:.2f} raw={r.raw_a_precision:.2f}")
        lines.append(f"         ctx_b: enc={r.encoder_b_precision:.2f} raw={r.raw_b_precision:.2f}")
        enc_total += r.encoder_avg
        raw_total += r.raw_avg
        if r.encoder_wins:
            enc_wins += 1

    n = len(results)
    lines.insert(1, f"Encoder avg: {enc_total/n:.3f}  Raw avg: {raw_total/n:.3f}  Encoder wins: {enc_wins}/{n}")
    return "\n".join(lines)
