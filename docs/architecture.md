# Architecture

## The Context Assembly Network

The core of cortex-net is a **Context Assembly Network (CAN)** — a set of small, trainable neural components that learn to compose the optimal context window for any given situation.

```
Input Situation → [Trainable Components] → Assembled Context → LLM → Action → Outcome
                                                                          ↓
                                                                   Outcome Signal
                                                                          ↓
                                                                 Update Components
```

The LLM remains frozen. It does what it's good at: reasoning, generation, tool use. The trainable components learn *what to show it* so it can do its best work.

## The Four Components

### 1. Situation Encoder

**What it does:** Builds a dense vector representation of "where am I right now?"

**Architecture:** 2-layer transformer or MLP

**Inputs:**
- Current message
- Recent conversation history
- Metadata: time of day, channel, user identity, task type, conversation length

**Output:** A situation embedding vector (e.g., 256-dim)

**Why it matters:** Every other component needs a shared understanding of the current situation. The Situation Encoder provides this as a learned representation, not a hand-crafted feature vector.

---

### 2. Memory Gate

**What it does:** Learns which stored memories are actually relevant to the current situation. Replaces naive cosine similarity with a trained relevance scorer.

**Architecture:** Bilinear scoring function

```
relevance(situation, memory) = situation · W · memory
```

Where `W` is a learned weight matrix.

**Inputs:**
- Situation embedding (from Situation Encoder)
- Candidate memory embeddings (from vector store)

**Output:** Relevance score per memory → top-k selection

**Training signal:** Did retrieving this memory lead to a better outcome? Measured by:
- Was the memory actually referenced in the response?
- Did the user confirm the information was helpful?
- Did the interaction succeed without correction?

**Why it matters:** This is the single highest-leverage component. Every RAG system in the world uses cosine similarity. A learned memory gate that understands *contextual* relevance — not just semantic similarity — would be a meaningful advance.

---

### 3. Strategy Selector

**What it does:** Learns which approach to take for different situations.

**Architecture:** Linear classification head + softmax over ~10-20 strategy profiles

**What's a strategy profile?** A combination of:
- Prompt framing (detailed analysis vs. quick answer vs. clarifying question)
- Tool permissions (research tools, code execution, none)
- Reasoning style (step-by-step vs. direct vs. exploratory)
- Response format (concise vs. thorough vs. structured)

**Example strategies:**

| Strategy | When | Framing |
|----------|------|---------|
| Deep Research | Complex factual question | "Think step by step, cite sources" |
| Quick Answer | Simple known question | "Be concise and direct" |
| Clarify First | Ambiguous request | "Ask one clarifying question" |
| Proactive Suggest | Pattern detected | "Surface the insight, suggest action" |
| Hedge & Escalate | Low confidence | "State uncertainty, offer alternatives" |

**Training signal:** Which strategy led to task success in similar situations.

**Why it matters:** Today, agents use the same approach for every situation. A human expert naturally shifts between modes — careful analysis, quick recall, asking questions, suggesting proactively. The Strategy Selector gives agents this flexibility, *learned* from experience.

---

### 4. Confidence Estimator

**What it does:** Predicts how confident the system should be in its current context and approach.

**Architecture:** 2-layer MLP → scalar output [0, 1]

**Inputs:**
- Situation embedding
- Summary of assembled context (how many relevant memories, strategy confidence, etc.)

**Output:** Predicted confidence score

**Behavior at different levels:**
- **High confidence (>0.8):** Proceed normally
- **Medium confidence (0.4-0.8):** Include caveats, offer to dig deeper
- **Low confidence (<0.4):** Ask for clarification, escalate, or explicitly state uncertainty

**Training signal:** Calibration against actual outcomes. The estimator should be well-calibrated: when it says 70% confident, it should be right ~70% of the time.

**Why it matters:** Current agents have no metacognition. They state wrong answers with the same confidence as right ones. A trained confidence estimator means the agent *knows when it doesn't know* — the foundation of trustworthiness.

---

## Parameter Sharing

All four components share the Situation Encoder's output. This means:
- The situation representation is trained by all downstream signals
- Total trainable parameters stay small (~1-10M)
- Components benefit from each other's learning

```
                    ┌→ Memory Gate ──────────────┐
                    │                             │
Situation Encoder ──┼→ Strategy Selector ─────────┼→ Context Assembly → LLM
                    │                             │
                    └→ Confidence Estimator ──────┘
```

## Training Signals

The system learns from three types of signal:

### Explicit Feedback
- User thumbs up/down
- Direct corrections ("that's wrong, actually...")
- Explicit preference statements

### Implicit Feedback
- Follow-up clarification needed → retrieved context was insufficient
- Task completed without errors → approach was right
- User said "thanks" / moved on → probably satisfied
- Conversation abandoned → probably failed

### Self-Evaluation
- After each interaction, the LLM rates its own performance (1-5)
- Cheap, fast, and surprisingly well-correlated with human judgment
- Used to bootstrap training before enough human signal accumulates

## Integration Points

cortex-net is designed to wrap any LLM and any memory system:

- **LLM:** Any model that accepts a text prompt and returns text. OpenAI, Anthropic, local models — doesn't matter.
- **Memory:** Any vector store that returns embeddings. cortex-memory, Pinecone, Chroma, Qdrant — the Memory Gate scores on top.
- **Tools:** Strategy profiles can include tool permission sets, but tool execution itself is handled by the LLM + host system.
