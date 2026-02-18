# Vision

## The Pieces Are All Here. The Mind Is Not.

We have very capable LLMs. We have mature RAG systems. We have tool-use, function calling, code execution, web browsing. Every piece of the puzzle exists. And yet — no one has built an agent that truly feels like a leap.

This is surprising. And it suggests the problem isn't capability. It's abstraction.

## The Wrong Abstraction

The industry has converged on a single pattern for AI agents:

```
LLM + Loop + Tools = Agent
```

Take a language model. Put it in a while loop. Give it tools. Call it an agent.

This is like putting an engine on a horse carriage and calling it a car. The architecture is wrong. You can improve the engine all day — it won't give you independent suspension.

Current "agents" are chatbots that can call functions. They have no persistent sense of self. No memory that learns. No ability to get better at their job. No metacognition. No genuine autonomy. Every context window is a fresh start — an amnesia patient reading their own diary.

## What a Real Agent Should Be

Strip away the industry hype. What properties should an intelligent agent actually have?

- **Coherent** — A unified sense of self that persists across interactions
- **Robust** — Graceful degradation when things go wrong, not chain failure
- **Self-aware** — Knows what it knows, what it doesn't, and when it's confused
- **Learning** — Gets better at its job over time, not just retrieves stored facts
- **Autonomous** — Genuine judgment, not just unsupervised execution
- **Proactive** — Initiates action based on understanding, not just responds to requests
- **No patchy logic** — Intelligence from the model, not from hardcoded if/else chains
- **Robust memory** — Consolidates, forgets noise, builds understanding — not just stores and retrieves

These are properties of a mind, not a pipeline. And no amount of prompt engineering or tool-calling will produce them.

## The Insight

Here's what we noticed: an LLM's behavior at any moment is determined by **what enters the context window**. The system prompt, the memories retrieved, the examples surfaced, the framing of the task — these are the real controls.

Current systems assemble this context with hand-coded rules. cosine similarity for memory retrieval. Fixed system prompts. Static tool lists.

**What if the context assembly layer could learn?**

Not the LLM — keep that frozen. It's already good at reasoning. But the layer that decides *what the LLM sees* — that's where intelligence at the agent level lives. And that layer can be made of small, trainable neural networks.

This is cortex-net: a trainable meta-architecture where small neural components learn from experience to assemble optimal context around a frozen LLM. The agent improves over time not by changing its brain, but by learning how to use it.

## Design Principles

1. **The LLM is the mind. cortex-net is the environment.** We don't orchestrate thinking. We learn to provide the right context for thinking to happen well.

2. **Intelligence scales with the model.** A better LLM should mean a better agent, with zero changes to cortex-net. The meta-architecture adds zero IQ — it optimizes the *conditions* for intelligence.

3. **Small and trainable beats large and fixed.** 1-10M trainable parameters that learn from every interaction, rather than 100B frozen parameters that never change.

4. **Learning is continuous.** Not batch fine-tuning. The agent gets better with every conversation, every success, every failure.

5. **Honest about uncertainty.** The system knows when it doesn't know. This is a trained capability, not a prompt hack.
