# Live Agent

The `CortexAgent` is a complete agent that wraps cortex-net around any OpenAI-compatible LLM. It handles the full loop: context assembly → LLM call → feedback extraction → learning.

## Quick Start

```python
from cortex_net.agent import CortexAgent, AgentConfig

agent = CortexAgent(
    AgentConfig(
        model="MiniMax-M1",
        base_url="https://api.minimax.io/v1",
        state_dir="./my_agent",
    ),
    api_key="sk-..."
)

# Seed with knowledge
agent.add_memories([
    "We use Kubernetes for production deployments",
    "CI/CD runs on GitHub Actions",
])

# Chat — cortex-net assembles context, LLM responds
response = agent.chat("How should I deploy the new service?")

# Next message — feedback is automatically extracted
response = agent.chat("Thanks, that worked!")
# ↑ "Thanks" → positive signal → components learn
```

## What Happens on Each `chat()` Call

```
1. Extract feedback from this message about the PREVIOUS response
   └─ "Thanks!" → positive, "That's wrong" → negative
   
2. Encode the situation
   └─ message + history + metadata → Situation Encoder → 384-dim embedding
   
3. Retrieve relevant memories via Memory Gate
   └─ MemoryBank.retrieve() with learned scoring
   
4. Select strategy via Strategy Selector
   └─ 1 of 10 strategies (deep_research, quick_answer, etc.)
   
5. Estimate confidence via Confidence Estimator
   └─ proceed (≥0.8) / hedge (0.4-0.8) / escalate (<0.4)
   
6. Build system prompt
   └─ base prompt + selected memories + strategy framing + confidence hedging
   
7. Call LLM (OpenAI-compatible API)
   
8. Auto-memorize the exchange
   └─ Non-trivial messages and responses get stored in MemoryBank
   
9. Store state for next feedback cycle
```

## Configuration

```python
@dataclass
class AgentConfig:
    model: str = "MiniMax-M1"              # Any OpenAI-compatible model
    base_url: str = "https://api.minimax.io/v1"  # API endpoint
    max_tokens: int = 1024
    text_dim: int = 384                    # Embedding dimension
    embedding_model: str = "all-MiniLM-L6-v2"  # Sentence-transformer model
    max_memories: int = 1000
    retrieval_k: int = 5                   # Memories to retrieve per query
    online_learning: bool = True           # Enable continuous learning
    update_every: int = 5                  # Train every N interactions
    replay_capacity: int = 10000
    state_dir: str = "./state"
    system_prompt: str = "You are a helpful assistant..."
```

## Compatible LLM Providers

Any OpenAI-compatible API works. Just change `base_url` and `model`:

| Provider | base_url | model |
|----------|----------|-------|
| MiniMax | `https://api.minimax.io/v1` | `MiniMax-M1` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o` |
| Anthropic (via proxy) | varies | `claude-sonnet-4-20250514` |
| Local (Ollama) | `http://localhost:11434/v1` | `llama3` |
| Together AI | `https://api.together.xyz/v1` | `meta-llama/...` |

## Creating Agents

Agents live in `agents/<name>/` with a `config.json`:

```json
{
  "name": "Atlas",
  "description": "DevOps assistant",
  "model": "MiniMax-M1",
  "base_url": "https://api.minimax.io/v1",
  "system_prompt": "You are Atlas, a DevOps assistant...",
  "initial_memories": [
    "We use Kubernetes for deployments",
    "Monitoring: Prometheus + Grafana"
  ]
}
```

Run interactively:
```bash
python agents/run_agent.py atlas --api-key "sk-..."
```

Or single query:
```bash
python agents/run_agent.py atlas -k "sk-..." -q "How do I deploy?"
```

## State

All state persists in `agents/<name>/state/`:

```
state/
├── memory.db                  # SQLite: memories + embeddings + metadata
├── blobs/                     # Binary content (images, files)
├── situation_encoder.pt       # Component weights
├── memory_gate.pt
├── strategy_selector.pt
├── confidence_estimator.pt
├── conversation.json          # Recent conversation history
└── replay_buffer.jsonl        # Online learning experience buffer
```

Kill the agent, restart it — picks up exactly where it left off.

## Stats

```python
agent.stats()
# {
#   'conversation_turns': 26,
#   'memories': 32,
#   'replay_buffer': 11,
#   'online_updates': 3,
#   'ema_loss': 0.54,
#   'gate_trained': False,
# }
```
