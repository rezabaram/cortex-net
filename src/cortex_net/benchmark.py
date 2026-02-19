"""Realistic benchmark — test Memory Gate vs cosine on real text.

Creates scenarios that expose cosine similarity's weaknesses:
1. Topically similar but irrelevant (distractors)
2. Topically different but contextually relevant
3. Nuanced relevance that requires learned judgment

Each scenario has: query, memory pool, ground truth relevant indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F

from cortex_net.embedding_store import EmbeddingStore
from cortex_net.eval import precision_at_k, recall_at_k
from cortex_net.memory_gate import MemoryGate


@dataclass
class Scenario:
    """A single benchmark scenario."""

    name: str
    query: str
    memories: list[str]
    relevant_indices: set[int]
    description: str = ""


@dataclass
class ScenarioResult:
    """Results for one scenario."""

    name: str
    gate_precision: float
    gate_recall: float
    cosine_precision: float
    cosine_recall: float
    k: int
    gate_ranking: list[int]
    cosine_ranking: list[int]

    @property
    def gate_wins(self) -> bool:
        return self.gate_precision > self.cosine_precision


@dataclass
class BenchmarkReport:
    """Full benchmark report."""

    results: list[ScenarioResult] = field(default_factory=list)

    @property
    def num_scenarios(self) -> int:
        return len(self.results)

    @property
    def gate_win_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.gate_wins) / len(self.results)

    @property
    def avg_gate_precision(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.gate_precision for r in self.results) / len(self.results)

    @property
    def avg_cosine_precision(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.cosine_precision for r in self.results) / len(self.results)

    def summary(self) -> str:
        lines = [
            f"Benchmark: {self.num_scenarios} scenarios",
            f"Gate  avg P@k: {self.avg_gate_precision:.3f}",
            f"Cosine avg P@k: {self.avg_cosine_precision:.3f}",
            f"Gate win rate: {self.gate_win_rate:.1%}",
            "",
        ]
        for r in self.results:
            winner = "GATE" if r.gate_wins else ("TIE" if r.gate_precision == r.cosine_precision else "COS")
            lines.append(
                f"  [{winner}] {r.name}: gate={r.gate_precision:.2f} cos={r.cosine_precision:.2f}"
            )
        return "\n".join(lines)


# ── Benchmark scenarios ──────────────────────────────────────────────

def build_scenarios() -> list[Scenario]:
    """Build realistic benchmark scenarios."""
    return [
        Scenario(
            name="distractor_trap",
            description="Topically similar memories that are NOT relevant, plus a relevant one from a different topic",
            query="How do I fix the authentication timeout error in our API?",
            memories=[
                # Relevant (contextually useful)
                "Last week we increased the JWT token expiry from 15min to 1hr after users reported session drops during long form submissions.",
                "The load balancer health check interval was changed to 30s which caused stale connections to the auth service.",
                # Distractors (topically similar but not helpful)
                "Authentication is the process of verifying the identity of a user or system.",
                "OAuth 2.0 is an authorization framework that enables applications to obtain limited access.",
                "API rate limiting helps prevent abuse by restricting the number of requests per time period.",
                "REST APIs use HTTP methods like GET, POST, PUT, and DELETE for resource operations.",
                # Irrelevant
                "The company picnic is scheduled for next Friday in the park.",
                "Python 3.12 introduced several new typing features.",
            ],
            relevant_indices={0, 1},
        ),
        Scenario(
            name="cross_domain_relevance",
            description="Relevant memory from an unrelated domain",
            query="Why is our ML model's accuracy dropping on weekends?",
            memories=[
                # Relevant (different domain but contextually key)
                "The data pipeline switches to a backup server on weekends with 2hr sync delay, so training data may be stale.",
                "Weekend traffic patterns differ significantly: 60% mobile vs 30% mobile on weekdays.",
                # Distractors (ML-related but not helpful)
                "Random forests are ensemble methods that combine multiple decision trees.",
                "Accuracy is calculated as the number of correct predictions divided by total predictions.",
                "Model overfitting occurs when the model learns noise in the training data.",
                "Cross-validation helps estimate how well a model generalizes to unseen data.",
                # Irrelevant
                "The office WiFi password was changed last Tuesday.",
                "Team standup is at 9:30 AM every day.",
            ],
            relevant_indices={0, 1},
        ),
        Scenario(
            name="temporal_context",
            description="Recent context matters more than semantic similarity",
            query="Should I deploy the payment service update today?",
            memories=[
                # Relevant (contextual warnings)
                "Deployment freeze announced for Feb 15-20 due to end-of-quarter processing.",
                "The payment service had a P1 incident yesterday — root cause was a race condition in the retry logic that hasn't been fully patched.",
                # Distractors (deployment-related but not relevant to THIS decision)
                "Deployment best practices include blue-green deployments and canary releases.",
                "Continuous deployment automates the release process from code commit to production.",
                "Microservices architecture allows independent deployment of services.",
                "Docker containers provide consistent deployment environments across stages.",
                # Irrelevant
                "The new coffee machine arrived in the kitchen.",
                "Q3 revenue targets were discussed in the all-hands meeting.",
            ],
            relevant_indices={0, 1},
        ),
        Scenario(
            name="negative_experience",
            description="Past failure is relevant even if topically distant",
            query="Can we use Redis for storing user sessions?",
            memories=[
                # Relevant (past experience)
                "We tried Redis for session storage in 2024 but hit memory limits at 50k concurrent users — switched to DynamoDB.",
                "Redis cluster failover caused a 12-minute outage last March when a primary node went down during peak hours.",
                # Distractors
                "Redis is an in-memory data structure store used as a database, cache, and message broker.",
                "Session management is crucial for maintaining user state across HTTP requests.",
                "NoSQL databases offer flexible schemas and horizontal scaling capabilities.",
                "Caching strategies include write-through, write-behind, and cache-aside patterns.",
                # Irrelevant
                "The design team is switching from Figma to Sketch.",
                "Annual performance reviews are due by end of month.",
            ],
            relevant_indices={0, 1},
        ),
        Scenario(
            name="subtle_preference",
            description="User preference matters, not just topic match",
            query="What format should I use for the weekly report?",
            memories=[
                # Relevant (user preferences)
                "Reza mentioned he prefers bullet points over paragraphs for status updates — says he skims them on mobile.",
                "Last time I sent a detailed report, the feedback was 'too long, just give me the highlights and blockers'.",
                # Distractors
                "Report writing best practices include clear headings and executive summaries.",
                "Markdown is a lightweight markup language for creating formatted text.",
                "Weekly status reports help teams track progress and identify blockers.",
                "Data visualization tools like charts and graphs enhance report readability.",
                # Irrelevant
                "The printer on floor 3 is out of toner.",
                "New hire orientation is every Monday at 10 AM.",
            ],
            relevant_indices={0, 1},
        ),
    ]


def run_benchmark(
    gate: MemoryGate,
    store: EmbeddingStore,
    scenarios: list[Scenario] | None = None,
    k: int = 3,
) -> BenchmarkReport:
    """Run the full benchmark.

    Args:
        gate: Trained (or untrained) Memory Gate.
        store: Embedding store for encoding texts.
        scenarios: Scenarios to run (defaults to built-in set).
        k: Number of results to evaluate.

    Returns:
        BenchmarkReport with per-scenario and aggregate results.
    """
    if scenarios is None:
        scenarios = build_scenarios()

    report = BenchmarkReport()

    for scenario in scenarios:
        query_emb = store.encode(scenario.query)
        memory_embs = store.encode_batch(scenario.memories)

        # Memory Gate ranking
        with torch.no_grad():
            gate_idx, _ = gate.select_top_k(query_emb, memory_embs, k=k)
        gate_ranking = gate_idx.cpu().tolist()

        # Cosine ranking
        q_norm = F.normalize(query_emb.unsqueeze(0), dim=-1)
        m_norm = F.normalize(memory_embs, dim=-1)
        cosine_scores = (q_norm @ m_norm.T).squeeze(0)
        cosine_ranking = torch.argsort(cosine_scores, descending=True)[:k].cpu().tolist()

        result = ScenarioResult(
            name=scenario.name,
            gate_precision=precision_at_k(
                torch.tensor(gate_ranking), scenario.relevant_indices, k
            ),
            gate_recall=recall_at_k(
                torch.tensor(gate_ranking), scenario.relevant_indices, k
            ),
            cosine_precision=precision_at_k(
                torch.tensor(cosine_ranking), scenario.relevant_indices, k
            ),
            cosine_recall=recall_at_k(
                torch.tensor(cosine_ranking), scenario.relevant_indices, k
            ),
            k=k,
            gate_ranking=gate_ranking,
            cosine_ranking=cosine_ranking,
        )
        report.results.append(result)

    return report


def train_on_scenarios(
    gate: MemoryGate,
    optimizer: torch.optim.Optimizer,
    store: EmbeddingStore,
    scenarios: list[Scenario] | None = None,
    epochs: int = 100,
    margin: float = 1.0,
) -> list[float]:
    """Train the Memory Gate on benchmark scenarios.

    Uses the ground truth labels from scenarios as training signal.

    Returns:
        List of average loss per epoch.
    """
    if scenarios is None:
        scenarios = build_scenarios()

    # Pre-encode everything
    encoded: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for s in scenarios:
        q = store.encode(s.query)
        mems = store.encode_batch(s.memories)
        pos_idx = list(s.relevant_indices)
        neg_idx = [i for i in range(len(s.memories)) if i not in s.relevant_indices]
        pos = mems[pos_idx]
        neg = mems[neg_idx]
        encoded.append((q, pos, neg))

    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for q, pos, neg in encoded:
            loss = gate.train_step(optimizer, q, pos, neg, margin=margin)
            total_loss += loss
        epoch_losses.append(total_loss / len(encoded))

    return epoch_losses
