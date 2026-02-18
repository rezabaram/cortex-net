# cortex-net

[![Status: Early Research](https://img.shields.io/badge/status-early%20research-orange)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Trainable meta-architecture for intelligent agents.**

Small neural networks that learn to assemble optimal context around frozen LLMs. The agent gets better at its job over time — not by changing the model, but by learning how to use it.

## The Problem

The industry has all the pieces — capable LLMs, good RAG, solid tooling — yet no one has built a truly intelligent agent. Current agents are "LLM + loop + tools." That's a chatbot that calls functions, not a mind.

## The Idea

Keep the LLM frozen. Wrap it in small, trainable neural components (~1-10M parameters) that learn from experience:

- **Memory Gate** — Learns what memories are actually relevant (replaces naive cosine similarity)
- **Situation Encoder** — Builds a representation of "where am I right now"
- **Strategy Selector** — Learns which approach works for which situation
- **Confidence Estimator** — Knows when it's likely wrong

Together, these form a **Context Assembly Network** that learns to compose the optimal context window for any given situation.

```
Situation → [Trainable Components] → Assembled Context → LLM → Action → Outcome
                                                                    ↓
                                                             Update Components
```

## Documentation

Full docs: [docs/](docs/) (or run `mkdocs serve`)

- [Vision](docs/vision.md) — Why this matters
- [Architecture](docs/architecture.md) — How it works
- [Implementation Plan](docs/implementation.md) — Phased roadmap

## Project Structure

```
src/cortex_net/
├── context_assembler.py    # Main pipeline
├── memory_gate.py          # Learned memory relevance scoring
├── situation_encoder.py    # Situation embedding
├── strategy_selector.py    # Strategy classification
└── confidence_estimator.py # Confidence calibration
```

## Status

Early research phase. Starting with the Memory Gate as proof of concept.

## License

MIT
