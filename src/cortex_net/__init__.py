"""cortex-net: Trainable meta-architecture for LLM context assembly."""

__version__ = "0.2.0"

from cortex_net.agent import CortexAgent, AgentConfig
from cortex_net.memory_bank import MemoryBank
from cortex_net.memory_gate import MemoryGate
from cortex_net.situation_encoder import SituationEncoder
from cortex_net.strategy_selector import StrategySelector, StrategyRegistry
from cortex_net.confidence_estimator import ConfidenceEstimator
from cortex_net.tools import ToolRegistry

__all__ = [
    "CortexAgent", "AgentConfig",
    "MemoryBank", "MemoryGate",
    "SituationEncoder",
    "StrategySelector", "StrategyRegistry",
    "ConfidenceEstimator",
    "ToolRegistry",
]
