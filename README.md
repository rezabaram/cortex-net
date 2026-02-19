# cortex-net

[![Tests](https://img.shields.io/badge/tests-102%20passing-brightgreen)]()
[![Parameters](https://img.shields.io/badge/params-1.1M-blue)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Trainable meta-architecture for intelligent agents.**

Small neural networks (~1.1M params) that learn to assemble optimal context around frozen LLMs. The agent gets better over time — not by changing the model, but by learning how to use it.

## Results

| System | Precision@3 | Recall@3 | Strategy | Confidence |
|--------|------------|----------|----------|------------|
| **cortex-net** | **0.833** | **0.875** | **100%** | **100%** |
| cosine RAG | 0.778 | 0.792 | — | — |
| no memory | 0.000 | 0.000 | — | — |

cortex-net outperforms standard RAG retrieval while adding strategy selection and confidence calibration that RAG doesn't have.

## Architecture

Four trainable components wrap a frozen LLM:

```
Query → Situation Encoder → ┬→ Memory Gate        → relevant memories
                            ├→ Strategy Selector   → approach profile
                            └→ Confidence Estimator → proceed/hedge/escalate
                                      ↓
                              Assembled Context → LLM → Action → Outcome
                                                                    ↓
                                                             Update Components
```

| Component | What it learns | Params |
|-----------|---------------|--------|
| **Situation Encoder** | Fuses text + history + metadata → situation embedding | 858K |
| **Memory Gate** | Which memories are relevant (replaces naive cosine similarity) | 147K |
| **Strategy Selector** | Which approach works for which situation (configurable sets: generic/developer/support) | 51K |
| **Confidence Estimator** | When it's likely wrong (calibrated, ECE < 0.01) | 50K |

All components train jointly with shared gradients through the Situation Encoder.

## Quick Start

```python
from cortex_net.context_assembler import ContextAssembler

assembler = ContextAssembler(state_dir="./state")

memories = [
    "We use Kubernetes for production deployments",
    "Last deploy caused a 10min outage from missing env vars",
    "Team lunch is on Fridays",
]

result = assembler.assemble(
    query="How should I deploy the new service?",
    memories=memories,
    k=2,
    metadata={"hour_of_day": 10, "conversation_length": 3},
)

print(result.prompt_prefix)
# → Selected memories, strategy, confidence framing ready for LLM
```

## Training

```python
from cortex_net.joint_trainer import JointTrainer, TrainingSample

trainer = JointTrainer(encoder, gate, selector, estimator, registry)
metrics = trainer.train(samples, epochs=100)
# All losses decrease jointly; gradients flow through Situation Encoder
```

## Installation

```bash
pip install -e .
```

Requires: Python 3.10+, PyTorch, sentence-transformers

## Project Structure

```
src/cortex_net/
├── context_assembler.py      # Full pipeline orchestration
├── memory_gate.py            # Bilinear memory relevance scoring
├── situation_encoder.py      # MLP: text + history + metadata → embedding
├── strategy_selector.py      # Categorical + continuous strategy selection
├── tools.py                  # Tool registry (file, shell, edit)
├── confidence_estimator.py   # Calibrated confidence (BCE + ECE)
├── joint_trainer.py          # End-to-end multi-task training
├── state_manager.py          # Atomic checkpointing, auto-resume
├── interaction_log.py        # JSONL interaction logging
├── embedding_store.py        # Persistent embedding cache
├── retrieval_pipeline.py     # End-to-end retrieval pipeline
├── eval.py                   # Precision@k, recall@k metrics
├── benchmark.py              # Phase 1 memory gate scenarios
├── situation_benchmark.py    # Phase 2 contextual scenarios
├── strategy_benchmark.py     # Phase 3 strategy evaluation
└── comparison_benchmark.py   # Full system comparison
```

## Key Design Decisions

- **Unified dimensions** (situation = text = 384): simplifies cosine fallback, identity init
- **MLP over Transformer** for Situation Encoder: sufficient for the fusion task, trains faster
- **Contrastive loss** for memory gate, **calibration loss** (BCE + soft ECE) for confidence
- **Atomic writes** for all state — crash-safe, resumable
- **Identity initialization** for Memory Gate W: untrained ≈ cosine similarity (smooth cold start)

## Documentation

Full docs: [docs/](docs/) or live at `http://localhost:8000/cortex-net/`

- [Vision](docs/vision.md) — Why this matters
- [Architecture](docs/architecture.md) — How it works
- [Implementation Plan](docs/implementation.md) — Phased roadmap with results

## License

MIT
