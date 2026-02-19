"""Strategy Selector benchmark — prove learned selection beats fixed strategy.

Creates scenarios where the correct strategy depends on the situation.
Measures: accuracy, diversity, and whether at least 5 strategies are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from cortex_net.situation_encoder import SituationEncoder, extract_metadata_features
from cortex_net.strategy_selector import StrategySelector, StrategyRegistry


@dataclass
class StrategyScenario:
    """A situation that maps to a specific best strategy."""

    name: str
    description: str
    message: str
    metadata: dict
    best_strategy_id: str
    history: list[str] = field(default_factory=list)


def build_strategy_scenarios() -> list[StrategyScenario]:
    return [
        StrategyScenario(
            name="complex_factual",
            description="Needs deep research",
            message="What are the performance implications of switching from PostgreSQL to CockroachDB for our distributed transaction workload?",
            metadata={"hour_of_day": 10, "conversation_length": 2, "message_length": 120},
            best_strategy_id="deep_research",
        ),
        StrategyScenario(
            name="simple_known",
            description="Simple factual question",
            message="What port does Redis run on by default?",
            metadata={"hour_of_day": 14, "conversation_length": 1, "message_length": 40},
            best_strategy_id="quick_answer",
        ),
        StrategyScenario(
            name="ambiguous_request",
            description="Needs clarification",
            message="Can you fix the thing?",
            metadata={"hour_of_day": 11, "conversation_length": 1, "message_length": 22},
            best_strategy_id="clarify_first",
        ),
        StrategyScenario(
            name="pattern_detected",
            description="User keeps asking same type of question",
            message="How do I check the logs again?",
            metadata={"hour_of_day": 15, "conversation_length": 30, "message_length": 32},
            history=["How do I check the logs?", "Where are the logs?", "Can you show me the log output?"],
            best_strategy_id="step_by_step",
        ),
        StrategyScenario(
            name="uncertain_domain",
            description="Question outside known domain",
            message="What's the best approach for quantum error correction in our system?",
            metadata={"hour_of_day": 16, "conversation_length": 5, "message_length": 65},
            best_strategy_id="deep_research",
        ),
        StrategyScenario(
            name="multi_step_task",
            description="Needs step-by-step execution",
            message="Set up a new microservice with Docker, CI/CD pipeline, and monitoring",
            metadata={"hour_of_day": 9, "conversation_length": 3, "message_length": 70},
            best_strategy_id="step_by_step",
        ),
        StrategyScenario(
            name="creative_task",
            description="Open-ended creative work",
            message="Help me brainstorm names for our new developer platform",
            metadata={"hour_of_day": 14, "conversation_length": 2, "message_length": 55},
            best_strategy_id="creative",
        ),
        StrategyScenario(
            name="code_help",
            description="Programming assistance",
            message="Write a Python function to merge two sorted linked lists",
            metadata={"hour_of_day": 11, "conversation_length": 1, "message_length": 55},
            best_strategy_id="step_by_step",
        ),
        StrategyScenario(
            name="info_overload",
            description="Too much information, needs distilling",
            message="Summarize the key decisions from today's 2-hour architecture review meeting",
            metadata={"hour_of_day": 17, "conversation_length": 1, "message_length": 70},
            best_strategy_id="summarize",
        ),
        StrategyScenario(
            name="frustrated_user",
            description="User expressing frustration",
            message="I've been trying to fix this bug for 3 hours and nothing works. I'm going crazy.",
            metadata={"hour_of_day": 22, "conversation_length": 15, "message_length": 80},
            best_strategy_id="empathize",
        ),
    ]


@dataclass
class StrategyBenchmarkResult:
    """Results of the strategy benchmark."""

    accuracy: float
    diversity_score: float
    num_strategies_used: int
    per_scenario: dict[str, tuple[str, str]]  # name → (predicted, actual)
    fixed_accuracy: float  # baseline: always pick the most common strategy

    def summary(self) -> str:
        lines = [
            f"Strategy Benchmark: accuracy={self.accuracy:.1%}, diversity={self.diversity_score:.2f}",
            f"Strategies used: {self.num_strategies_used}/10",
            f"vs fixed baseline: {self.fixed_accuracy:.1%}",
            "",
        ]
        for name, (pred, actual) in self.per_scenario.items():
            match = "✓" if pred == actual else "✗"
            lines.append(f"  [{match}] {name}: predicted={pred}, actual={actual}")
        return "\n".join(lines)


def train_strategy_selector(
    encoder: SituationEncoder,
    selector: StrategySelector,
    registry: StrategyRegistry,
    scenarios: list[StrategyScenario] | None = None,
    epochs: int = 200,
    lr: float = 1e-3,
    text_encoder=None,
    msg_embs: dict | None = None,
    hist_embs: dict | None = None,
) -> list[float]:
    """Train the strategy selector on labeled scenarios."""
    if scenarios is None:
        scenarios = build_strategy_scenarios()

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(selector.parameters()), lr=lr
    )

    # Pre-compute text embeddings (or use provided)
    device = next(encoder.parameters()).device
    if msg_embs is not None and hist_embs is not None:
        pass  # use provided embeddings
    elif text_encoder is not None:
        msg_embs = {s.name: text_encoder.encode(s.message, convert_to_tensor=True).to(device) for s in scenarios}
        hist_embs = {}
        for s in scenarios:
            if s.history:
                h = text_encoder.encode(s.history, convert_to_tensor=True).to(device).mean(dim=0)
            else:
                h = torch.zeros(encoder.text_dim, device=device)
            hist_embs[s.name] = h
    else:
        # Use random embeddings for testing without text encoder
        msg_embs = {s.name: torch.randn(encoder.text_dim, device=device) for s in scenarios}
        hist_embs = {s.name: torch.randn(encoder.text_dim, device=device) for s in scenarios}

    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        for s in scenarios:
            meta = extract_metadata_features(s.metadata, s.message, s.history).to(device)
            sit = encoder(msg_embs[s.name], hist_embs[s.name], meta)

            target_idx = registry.id_to_index(s.best_strategy_id)
            if target_idx is None:
                continue

            loss = selector.strategy_loss(sit, target_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_losses.append(total_loss / len(scenarios))

    return epoch_losses


def run_strategy_benchmark(
    encoder: SituationEncoder,
    selector: StrategySelector,
    registry: StrategyRegistry,
    scenarios: list[StrategyScenario] | None = None,
    text_encoder=None,
    msg_embs: dict | None = None,
    hist_embs: dict | None = None,
) -> StrategyBenchmarkResult:
    """Evaluate strategy selection accuracy."""
    if scenarios is None:
        scenarios = build_strategy_scenarios()

    device = next(encoder.parameters()).device

    if msg_embs is None:
        if text_encoder is not None:
            msg_embs = {s.name: text_encoder.encode(s.message, convert_to_tensor=True).to(device) for s in scenarios}
        else:
            msg_embs = {s.name: torch.randn(encoder.text_dim, device=device) for s in scenarios}

    if hist_embs is None:
        if text_encoder is not None:
            hist_embs = {}
            for s in scenarios:
                if s.history:
                    h = text_encoder.encode(s.history, convert_to_tensor=True).to(device).mean(dim=0)
                else:
                    h = torch.zeros(encoder.text_dim, device=device)
                hist_embs[s.name] = h
        else:
            hist_embs = {s.name: torch.randn(encoder.text_dim, device=device) for s in scenarios}

    correct = 0
    per_scenario = {}
    strategies_used = set()

    for s in scenarios:
        meta = extract_metadata_features(s.metadata, s.message, s.history).to(device)
        with torch.no_grad():
            sit = encoder(msg_embs[s.name], hist_embs[s.name], meta)
            selection = selector.select(sit, registry, explore=False)

        strategies_used.add(selection.strategy_id)
        is_correct = selection.strategy_id == s.best_strategy_id
        if is_correct:
            correct += 1
        per_scenario[s.name] = (selection.strategy_id, s.best_strategy_id)

    # Fixed baseline: always pick the most common strategy
    from collections import Counter
    strategy_counts = Counter(s.best_strategy_id for s in scenarios)
    most_common = strategy_counts.most_common(1)[0][1]
    fixed_accuracy = most_common / len(scenarios)

    return StrategyBenchmarkResult(
        accuracy=correct / len(scenarios),
        diversity_score=selector.diversity_score(),
        num_strategies_used=len(strategies_used),
        per_scenario=per_scenario,
        fixed_accuracy=fixed_accuracy,
    )
