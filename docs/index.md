# cortex-net

## Trainable Meta-Architecture for Intelligent Agents

**cortex-net** is a research project exploring a simple but underexplored idea: what if the layer around the LLM could *learn*?

Current AI agents treat the LLM as a black box inside a hand-coded pipeline. The pipeline decides what context to include, which tools to offer, how to frame the task. When it works, it works. When it doesn't, you add more if/else branches and hope for the best.

cortex-net takes a different approach. The LLM stays frozen — it's already good at reasoning. But the **context assembly layer** around it becomes a set of small, trainable neural networks that learn from every interaction:

- What memories actually matter for this situation
- Which strategy will work best right now
- How confident the system should be in its answer
- How to represent the current situation for all of the above

The result: an agent that gets meaningfully better at its job over time, without fine-tuning the underlying model.

## Quick Links

- [Vision](vision.md) — The problem we're solving and why it matters
- [Architecture](architecture.md) — The Context Assembly Network in detail
- [Why This Matters](why.md) — What's broken in today's agents
- [Implementation Plan](implementation.md) — Phased roadmap from prototype to integration

## Status

✅ **Phase 1** — Memory Gate (+67% vs cosine)
✅ **Phase 2** — Situation Encoder (+50% on contextual retrieval)
✅ **Phase 3** — Strategy Selector (10 profiles, learned selection)
✅ **Phase 4** — Confidence Estimator (ECE = 0.01)
✅ **Phase 5** — Context Assembler (full pipeline, 811K params)
📊 **94 tests passing** — See [Implementation Plan](implementation.md).
