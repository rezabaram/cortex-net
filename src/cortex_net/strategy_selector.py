"""Strategy Selector — learns which approach works for which situation.

Two modes:
1. **Categorical** (original): MLP classifier over fixed strategy profiles.
   Good for clear-cut domains with known strategies.

2. **Continuous** (new): MLP outputs a strategy embedding in a learned space.
   Strategies are soft — the model outputs *how* to respond rather than
   picking from a fixed menu. Strategy presets are anchor points in this space,
   not hard boundaries.

The continuous mode also supports predefined strategy sets for different
agent types (developer, support, creative, etc.).

Training signal: which strategy led to task success in similar past situations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Strategy Profiles ────────────────────────────────────────────────

@dataclass
class StrategyProfile:
    """A named strategy with its configuration."""

    id: str
    name: str
    description: str
    prompt_framing: str = ""
    reasoning_style: str = "direct"  # "direct" | "step_by_step" | "exploratory"
    response_format: str = "concise"  # "concise" | "thorough" | "structured"
    tool_permissions: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


# ── Predefined Strategy Sets ─────────────────────────────────────────

STRATEGY_SETS: dict[str, list[StrategyProfile]] = {}


def _register_set(name: str, strategies: list[StrategyProfile]) -> None:
    STRATEGY_SETS[name] = strategies


# --- Generic (backward-compatible default) ---
_register_set("generic", [
    StrategyProfile(
        id="deep_research", name="Deep Research",
        description="Complex factual question needing thorough analysis",
        prompt_framing="Think step by step. Cite sources. Be thorough.",
        reasoning_style="step_by_step", response_format="thorough",
        tool_permissions=["web_search", "code_exec"],
    ),
    StrategyProfile(
        id="quick_answer", name="Quick Answer",
        description="Simple known question, respond concisely",
        prompt_framing="Be concise and direct. No unnecessary explanation.",
        reasoning_style="direct", response_format="concise",
    ),
    StrategyProfile(
        id="clarify_first", name="Clarify First",
        description="Ambiguous request, ask one clarifying question",
        prompt_framing="Ask one specific clarifying question before proceeding.",
        reasoning_style="direct", response_format="concise",
    ),
    StrategyProfile(
        id="step_by_step", name="Step by Step",
        description="Multi-step task needing structured execution",
        prompt_framing="Break this into numbered steps. Execute each step carefully.",
        reasoning_style="step_by_step", response_format="structured",
        tool_permissions=["code_exec"],
    ),
    StrategyProfile(
        id="summarize", name="Summarize",
        description="Distill information into key points",
        prompt_framing="Extract the key points. Be brief and actionable.",
        reasoning_style="direct", response_format="structured",
    ),
    StrategyProfile(
        id="creative", name="Creative",
        description="Open-ended creative task, explore possibilities",
        prompt_framing="Explore multiple angles. Be creative and generative.",
        reasoning_style="exploratory", response_format="thorough",
    ),
    StrategyProfile(
        id="empathize", name="Empathize & Support",
        description="User expressing frustration or confusion, be supportive",
        prompt_framing="Acknowledge the difficulty. Be supportive. Then help.",
        reasoning_style="direct", response_format="concise",
    ),
])


# --- Developer ---
_register_set("developer", [
    StrategyProfile(
        id="implement", name="Implement",
        description="Write new code — feature, function, module, or class",
        prompt_framing="Write clean, working code. Include types and docstrings. Show the full implementation.",
        reasoning_style="step_by_step", response_format="structured",
        tool_permissions=["file_read", "file_write", "file_edit", "shell"],
    ),
    StrategyProfile(
        id="debug", name="Debug",
        description="Diagnose and fix a bug — trace the issue, find root cause, apply fix",
        prompt_framing="Reproduce the issue. Trace to root cause. Fix it. Verify the fix.",
        reasoning_style="step_by_step", response_format="structured",
        tool_permissions=["file_read", "file_edit", "shell"],
    ),
    StrategyProfile(
        id="refactor", name="Refactor",
        description="Improve code structure without changing behavior",
        prompt_framing="Identify the smell. Refactor incrementally. Run tests after each change.",
        reasoning_style="step_by_step", response_format="structured",
        tool_permissions=["file_read", "file_edit", "shell"],
    ),
    StrategyProfile(
        id="review", name="Code Review",
        description="Review code for correctness, style, performance, and security",
        prompt_framing="Check correctness first, then style, performance, security. Be specific about what to change and why.",
        reasoning_style="step_by_step", response_format="thorough",
        tool_permissions=["file_read", "shell"],
    ),
    StrategyProfile(
        id="test", name="Write Tests",
        description="Write or improve test coverage",
        prompt_framing="Cover happy path, edge cases, and error cases. Use descriptive test names.",
        reasoning_style="step_by_step", response_format="structured",
        tool_permissions=["file_read", "file_write", "shell"],
    ),
    StrategyProfile(
        id="explain", name="Explain Code",
        description="Explain how code works — architecture, flow, or specific logic",
        prompt_framing="Walk through the code. Explain the why, not just the what. Use examples.",
        reasoning_style="step_by_step", response_format="thorough",
        tool_permissions=["file_read"],
    ),
    StrategyProfile(
        id="architect", name="Architect",
        description="Design a system, module, or API — think about structure before coding",
        prompt_framing="Define the problem. Consider alternatives. Propose a design with clear trade-offs.",
        reasoning_style="exploratory", response_format="thorough",
        tool_permissions=["file_read", "file_list"],
    ),
    StrategyProfile(
        id="quick_fix", name="Quick Fix",
        description="Small, obvious fix — typo, import, config change",
        prompt_framing="Just fix it. Minimal explanation needed.",
        reasoning_style="direct", response_format="concise",
        tool_permissions=["file_read", "file_edit", "shell"],
    ),
    StrategyProfile(
        id="document", name="Document",
        description="Write or update documentation — README, docstrings, API docs",
        prompt_framing="Be clear and concise. Include examples. Write for the reader, not the author.",
        reasoning_style="direct", response_format="structured",
        tool_permissions=["file_read", "file_write", "file_edit"],
    ),
    StrategyProfile(
        id="explore", name="Explore Codebase",
        description="Navigate and understand an unfamiliar codebase or module",
        prompt_framing="Start from the entry point. Map the structure. Summarize what you find.",
        reasoning_style="exploratory", response_format="thorough",
        tool_permissions=["file_read", "file_list", "shell"],
    ),
    StrategyProfile(
        id="optimize", name="Optimize",
        description="Improve performance — profile, identify bottleneck, fix",
        prompt_framing="Measure first. Identify the bottleneck. Optimize with evidence.",
        reasoning_style="step_by_step", response_format="structured",
        tool_permissions=["file_read", "file_edit", "shell"],
    ),
    StrategyProfile(
        id="deploy", name="Deploy & DevOps",
        description="Deployment, CI/CD, infrastructure, configuration",
        prompt_framing="Be precise about commands and configs. Verify each step works.",
        reasoning_style="step_by_step", response_format="structured",
        tool_permissions=["file_read", "file_write", "file_edit", "shell"],
    ),
])


# --- Support ---
_register_set("support", [
    StrategyProfile(
        id="diagnose", name="Diagnose Issue",
        description="User reports a problem — gather info, identify cause",
        prompt_framing="Ask targeted questions. Narrow down the cause systematically.",
        reasoning_style="step_by_step", response_format="structured",
    ),
    StrategyProfile(
        id="guide", name="Step-by-Step Guide",
        description="Walk user through a procedure",
        prompt_framing="Number each step. Be precise. Confirm completion before proceeding.",
        reasoning_style="step_by_step", response_format="structured",
    ),
    StrategyProfile(
        id="escalate", name="Escalate",
        description="Issue beyond scope — escalate with context",
        prompt_framing="Summarize what was tried. State why escalation is needed. Provide all context.",
        reasoning_style="direct", response_format="structured",
    ),
    StrategyProfile(
        id="quick_answer", name="Quick Answer",
        description="Simple question with a known answer",
        prompt_framing="Answer directly. Link to docs if available.",
        reasoning_style="direct", response_format="concise",
    ),
    StrategyProfile(
        id="empathize", name="Empathize",
        description="User is frustrated — acknowledge and support",
        prompt_framing="Acknowledge the frustration. Be patient. Then help.",
        reasoning_style="direct", response_format="concise",
    ),
    StrategyProfile(
        id="clarify", name="Clarify",
        description="Request is unclear — ask for specifics",
        prompt_framing="Ask one specific question to disambiguate.",
        reasoning_style="direct", response_format="concise",
    ),
    StrategyProfile(
        id="workaround", name="Workaround",
        description="No direct fix available — offer alternative",
        prompt_framing="Explain why the direct approach won't work. Offer the best workaround.",
        reasoning_style="direct", response_format="structured",
    ),
])


# ── Helpers ──────────────────────────────────────────────────────────

def get_strategy_set(name: str) -> list[StrategyProfile]:
    """Get a predefined strategy set by name."""
    if name not in STRATEGY_SETS:
        available = ", ".join(STRATEGY_SETS.keys())
        raise ValueError(f"Unknown strategy set '{name}'. Available: {available}")
    return list(STRATEGY_SETS[name])  # return copy


def list_strategy_sets() -> list[str]:
    """List available predefined strategy sets."""
    return list(STRATEGY_SETS.keys())


def merge_strategy_sets(*names: str) -> list[StrategyProfile]:
    """Merge multiple strategy sets, deduplicating by id (last wins)."""
    seen: dict[str, StrategyProfile] = {}
    for name in names:
        for s in get_strategy_set(name):
            seen[s.id] = s
    return list(seen.values())


# ── Strategy Registry ────────────────────────────────────────────────

# Keep DEFAULT_STRATEGIES as alias for backward compat
DEFAULT_STRATEGIES = STRATEGY_SETS["generic"]


class StrategyRegistry:
    """Registry of available strategy profiles."""

    def __init__(self, strategies: list[StrategyProfile] | None = None) -> None:
        self.strategies = strategies or list(DEFAULT_STRATEGIES)
        self._id_to_idx = {s.id: i for i, s in enumerate(self.strategies)}

    @classmethod
    def from_set(cls, name: str) -> "StrategyRegistry":
        """Create a registry from a predefined strategy set."""
        return cls(get_strategy_set(name))

    @classmethod
    def from_sets(cls, *names: str) -> "StrategyRegistry":
        """Create a registry by merging multiple predefined sets."""
        return cls(merge_strategy_sets(*names))

    def __len__(self) -> int:
        return len(self.strategies)

    def __getitem__(self, idx: int) -> StrategyProfile:
        return self.strategies[idx]

    def get_by_id(self, strategy_id: str) -> StrategyProfile | None:
        idx = self._id_to_idx.get(strategy_id)
        return self.strategies[idx] if idx is not None else None

    def id_to_index(self, strategy_id: str) -> int | None:
        return self._id_to_idx.get(strategy_id)

    @property
    def ids(self) -> list[str]:
        return [s.id for s in self.strategies]

    def add(self, strategy: StrategyProfile) -> None:
        """Add a strategy dynamically."""
        if strategy.id in self._id_to_idx:
            # Replace existing
            idx = self._id_to_idx[strategy.id]
            self.strategies[idx] = strategy
        else:
            self._id_to_idx[strategy.id] = len(self.strategies)
            self.strategies.append(strategy)


# ── Continuous Strategy Space ─────────────────────────────────────────

@dataclass
class ContinuousStrategyOutput:
    """Output from continuous strategy selector.

    Instead of picking one strategy, outputs a point in strategy space.
    The nearest anchors and their distances describe the blend.
    """
    embedding: torch.Tensor           # (strategy_dim,) — the strategy point
    nearest_strategies: list[str]     # ordered by proximity
    weights: dict[str, float]         # how much each anchor contributes
    attributes: dict[str, float]      # interpolated attributes
    entropy: float = 0.0              # uncertainty measure — high = ambiguous
    certainty: bool = True            # True if entropy < threshold
    fallback: str | None = None       # "clarify" if uncertain, else None

    @property
    def primary(self) -> str:
        """Strongest strategy component."""
        return self.nearest_strategies[0] if self.nearest_strategies else "unknown"

    def prompt_framing(self, registry: StrategyRegistry) -> str:
        """Build a blended prompt framing from weighted strategies."""
        parts = []
        for sid, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            if w < 0.1:
                continue
            s = registry.get_by_id(sid)
            if s and s.prompt_framing:
                parts.append(s.prompt_framing)
        return " ".join(parts) if parts else ""


class ContinuousStrategySelector(nn.Module):
    """Strategy selector that outputs continuous strategy embeddings.

    Instead of classifying into N categories, projects situation into a
    strategy embedding space. Strategy presets are anchor points in this space.

    Architecture:
        situation (384) → Linear → ReLU → Linear → L2-norm → strategy_embedding (64)

    Strategy anchors are learned embeddings initialized from strategy descriptions.
    The model learns to place situations near the right strategy anchors.

    Args:
        situation_dim: Input dimension (default 384).
        strategy_dim: Strategy embedding dimension (default 64).
        hidden_dim: Hidden layer dimension (default 128).
    """

    def __init__(
        self,
        situation_dim: int = 384,
        strategy_dim: int = 64,
        hidden_dim: int = 128,
        num_anchors: int = 12,
    ) -> None:
        super().__init__()
        self.situation_dim = situation_dim
        self.strategy_dim = strategy_dim

        self.projector = nn.Sequential(
            nn.Linear(situation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, strategy_dim),
        )

        # Strategy anchors — learned positions in strategy space
        self.anchors = nn.Parameter(torch.randn(num_anchors, strategy_dim) * 0.1)
        self._anchor_ids: list[str] = []

        # Attribute heads — predict continuous strategy attributes
        # These let us interpolate between strategies smoothly
        self.attr_heads = nn.ModuleDict({
            "reasoning_depth": nn.Linear(strategy_dim, 1),    # 0=direct, 1=thorough
            "creativity": nn.Linear(strategy_dim, 1),          # 0=conservative, 1=exploratory
            "verbosity": nn.Linear(strategy_dim, 1),           # 0=concise, 1=verbose
            "tool_intensity": nn.Linear(strategy_dim, 1),      # 0=no tools, 1=heavy tool use
            "caution": nn.Linear(strategy_dim, 1),             # 0=confident, 1=cautious
        })

        self.register_buffer("_trained", torch.tensor(False))

    def init_anchors_from_registry(self, registry: StrategyRegistry) -> None:
        """Initialize anchor positions from strategy profiles.

        Maps strategy attributes to initial anchor positions so the space
        starts with meaningful structure.
        """
        n = min(len(registry), self.anchors.size(0))
        self._anchor_ids = [registry[i].id for i in range(n)]

        # Map strategy attributes to rough positions
        style_map = {"direct": 0.0, "step_by_step": 0.5, "exploratory": 1.0}
        format_map = {"concise": 0.0, "structured": 0.5, "thorough": 1.0}

        with torch.no_grad():
            for i in range(n):
                s = registry[i]
                # Use attributes to seed anchor positions
                depth = style_map.get(s.reasoning_style, 0.5)
                verbose = format_map.get(s.response_format, 0.5)
                tools = 1.0 if s.tool_permissions else 0.0
                # Spread anchors in a structured way
                self.anchors[i, 0] = depth
                self.anchors[i, 1] = verbose
                self.anchors[i, 2] = tools
                # Add some random spread in remaining dims
                self.anchors[i, 3:] = torch.randn(self.strategy_dim - 3) * 0.1

    @property
    def anchor_ids(self) -> list[str]:
        return self._anchor_ids

    def forward(self, situation: torch.Tensor) -> torch.Tensor:
        """Project situation to strategy space.

        Args:
            situation: (D,) or (B, D) situation embedding.

        Returns:
            L2-normalized strategy embedding: (strategy_dim,) or (B, strategy_dim).
        """
        emb = self.projector(situation)
        return F.normalize(emb, p=2, dim=-1)

    def select(
        self,
        situation: torch.Tensor,
        registry: StrategyRegistry,
        top_k: int = 3,
    ) -> ContinuousStrategyOutput:
        """Select strategy blend for a situation.

        Args:
            situation: (D,) situation embedding.
            registry: Strategy registry with anchor strategies.
            top_k: Number of nearest strategies to include.

        Returns:
            ContinuousStrategyOutput with embedding, nearest strategies, weights.
        """
        with torch.no_grad():
            emb = self.forward(situation)

            # Distance to each anchor
            n = min(len(self._anchor_ids), self.anchors.size(0))
            if n == 0:
                # No anchors initialized
                return ContinuousStrategyOutput(
                    embedding=emb,
                    nearest_strategies=[],
                    weights={},
                    attributes=self._predict_attributes(emb),
                )

            anchors_norm = F.normalize(self.anchors[:n], p=2, dim=-1)
            similarities = torch.mv(anchors_norm, emb)  # (n,)

            # Softmax over similarities for weights
            weights = F.softmax(similarities * 5.0, dim=0)  # temperature=0.2

            # Top-k
            top_vals, top_idx = torch.topk(weights, min(top_k, n))
            nearest = [self._anchor_ids[i] for i in top_idx.tolist()]
            weight_dict = {
                self._anchor_ids[i]: weights[i].item()
                for i in range(n)
                if weights[i].item() > 0.01
            }

            attrs = self._predict_attributes(emb)

        # Compute entropy and certainty from weight distribution
        weight_vals = weights[weights > 0]
        if weight_vals.numel() > 0:
            entropy = -(weight_vals * weight_vals.log()).sum()
            max_entropy = torch.tensor(float(n)).log()
            entropy_norm = (entropy / max_entropy).item() if max_entropy > 0 else 0.0
            certainty = entropy_norm < 0.5
            fallback = "clarify" if not certainty else None
        else:
            entropy_norm = 0.0
            certainty = True
            fallback = None

        return ContinuousStrategyOutput(
            embedding=emb,
            nearest_strategies=nearest,
            weights=weight_dict,
            attributes=attrs,
            entropy=entropy_norm,
            certainty=certainty,
            fallback=fallback,
        )

    def _predict_attributes(self, emb: torch.Tensor) -> dict[str, float]:
        """Predict continuous strategy attributes from embedding."""
        return {
            name: torch.sigmoid(head(emb)).item()
            for name, head in self.attr_heads.items()
        }

    def anchor_loss(
        self,
        situation: torch.Tensor,
        target_anchor_idx: int,
        margin: float = 0.3,
    ) -> torch.Tensor:
        """Contrastive loss: pull toward target anchor, push from others.

        Like the memory gate's training — bring the embedding close to the
        right anchor while maintaining distance from wrong ones.
        """
        emb = self.forward(situation.unsqueeze(0)).squeeze(0)  # (strategy_dim,)
        anchors_norm = F.normalize(self.anchors, p=2, dim=-1)

        # Positive: similarity to target
        pos_sim = torch.dot(emb, anchors_norm[target_anchor_idx])

        # Negatives: similarities to non-targets
        mask = torch.ones(self.anchors.size(0), dtype=torch.bool)
        mask[target_anchor_idx] = False
        neg_sims = torch.mv(anchors_norm[mask], emb)

        # Triplet-style loss: want pos_sim > max(neg_sims) + margin
        hardest_neg = neg_sims.max()
        loss = F.relu(hardest_neg - pos_sim + margin)
        return loss

    def attribute_loss(
        self,
        situation: torch.Tensor,
        target_attrs: dict[str, float],
    ) -> torch.Tensor:
        """MSE loss on predicted attributes."""
        emb = self.forward(situation.unsqueeze(0)).squeeze(0)
        loss = torch.tensor(0.0, device=situation.device)
        count = 0
        for name, target in target_attrs.items():
            if name in self.attr_heads:
                pred = torch.sigmoid(self.attr_heads[name](emb)).squeeze()
                loss = loss + F.mse_loss(pred, torch.tensor(target, device=situation.device))
                count += 1
        return loss / max(count, 1)


# ── Categorical Strategy Selector (original) ─────────────────────────

@dataclass
class StrategySelection:
    """Result of categorical strategy selection."""

    strategy_id: str
    strategy: StrategyProfile
    confidence: float
    probabilities: dict[str, float]
    entropy: float = 0.0          # uncertainty measure — high = ambiguous
    certainty: bool = True        # True if entropy < threshold
    fallback: str | None = None   # "clarify" if uncertain, else None


class StrategySelector(nn.Module):
    """Learned strategy selection from situation embeddings (categorical).

    Architecture:
        situation_embedding → Linear → ReLU → Linear → softmax → strategy distribution

    Args:
        situation_dim: Dimension of input situation embeddings.
        num_strategies: Number of strategy profiles.
        hidden_dim: Hidden layer dimension.
        exploration_rate: Probability of random strategy (epsilon-greedy).
        temperature: Softmax temperature (higher = more exploration).
    """

    def __init__(
        self,
        situation_dim: int = 256,
        num_strategies: int = 10,
        hidden_dim: int = 128,
        exploration_rate: float = 0.1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.situation_dim = situation_dim
        self.num_strategies = num_strategies
        self.exploration_rate = exploration_rate
        self.temperature = temperature

        self.classifier = nn.Sequential(
            nn.Linear(situation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_strategies),
        )

        self.register_buffer(
            "_usage_counts",
            torch.zeros(num_strategies, dtype=torch.long),
        )

    def forward(self, situation: torch.Tensor) -> torch.Tensor:
        return self.classifier(situation)

    def select(
        self,
        situation: torch.Tensor,
        registry: StrategyRegistry,
        explore: bool = True,
    ) -> StrategySelection:
        with torch.no_grad():
            logits = self.forward(situation)
            probs = F.softmax(logits / self.temperature, dim=-1)

            if explore and torch.rand(1).item() < self.exploration_rate:
                idx = torch.randint(0, self.num_strategies, (1,)).item()
            else:
                idx = torch.argmax(probs).item()

            self._usage_counts[idx] += 1

        strategy = registry[idx]
        prob_dict = {
            registry[i].id: probs[i].item()
            for i in range(min(len(registry), self.num_strategies))
        }

        return StrategySelection(
            strategy_id=strategy.id,
            strategy=strategy,
            confidence=probs[idx].item(),
            probabilities=prob_dict,
            entropy=self.compute_entropy(probs),
            certainty=self.compute_certainty(probs),
            fallback=self.compute_fallback(probs),
        )

    def compute_entropy(self, probs: torch.Tensor) -> float:
        """Compute entropy of probability distribution."""
        p = probs[probs > 0]
        if p.numel() == 0:
            return 0.0
        entropy = -(p * p.log()).sum()
        max_entropy = torch.tensor(float(self.num_strategies)).log()
        return (entropy / max_entropy).item() if max_entropy > 0 else 0.0

    def compute_certainty(self, probs: torch.Tensor, threshold: float = 0.5) -> bool:
        """Compute certainty based on entropy threshold."""
        entropy = self.compute_entropy(probs)
        return entropy < threshold

    def compute_fallback(self, probs: torch.Tensor) -> str | None:
        """Return fallback action if uncertain."""
        if not self.compute_certainty(probs):
            return "clarify"
        return None

    def strategy_loss(
        self,
        situation: torch.Tensor,
        target_strategy_idx: int,
    ) -> torch.Tensor:
        logits = self.forward(situation.unsqueeze(0))
        target = torch.tensor([target_strategy_idx], device=situation.device)
        return F.cross_entropy(logits, target)

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        situation: torch.Tensor,
        target_strategy_idx: int,
    ) -> float:
        optimizer.zero_grad()
        loss = self.strategy_loss(situation, target_strategy_idx)
        loss.backward()
        optimizer.step()
        return loss.item()

    def diversity_score(self) -> float:
        counts = self._usage_counts.float()
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        entropy = -(probs * probs.log()).sum()
        max_entropy = torch.tensor(float(self.num_strategies)).log()
        return (entropy / max_entropy).item() if max_entropy > 0 else 0.0

    def usage_summary(self, registry: StrategyRegistry) -> dict[str, int]:
        return {
            registry[i].id: self._usage_counts[i].item()
            for i in range(min(len(registry), self.num_strategies))
        }
