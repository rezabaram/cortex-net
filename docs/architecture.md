# Architecture

## The Context Assembly Network

cortex-net is a **Context Assembly Network (CAN)** — small, trainable neural components that learn to compose the optimal context window for a frozen LLM.

```
Query + History + Metadata
        ↓
  Situation Encoder (MLP, 858K params)
        ↓ situation embedding (384-dim)
        ├──→ Memory Gate (bilinear, 147K) ──→ relevant memories
        ├──→ Strategy Selector (MLP, 51K) ──→ approach profile
        └──→ Confidence Estimator (MLP, 50K) → proceed/hedge/escalate
                    ↓
            Assembled Context → LLM → Response
                                        ↓
                                  User Reaction
                                        ↓
                                Feedback Collector
                                        ↓
                              Online Trainer (replay buffer)
                                        ↓
                              Update all components
```

**Total: ~1.1M trainable parameters.** The LLM stays frozen.

---

## Components

### 1. Situation Encoder

Fuses three input streams into a single situation embedding that all downstream components share.

**Architecture:** 2-layer MLP with ReLU, L2-normalized output.

```
Input: concat(text_emb, history_emb, metadata_features) → (384 + 384 + 8 = 776-dim)
Hidden: Linear(776, 512) → ReLU → Dropout
Output: Linear(512, 384) → L2 normalize
```

**Metadata features** (8-dim): hour of day, day of week, is weekend, message length (log-scaled), history length (log-scaled), conversation length (log-scaled), question mark present, exclamation present.

**Text embeddings:** sentence-transformers `all-MiniLM-L6-v2` (384-dim). Same model for query, history, and memories.

**Training:** NT-Xent contrastive loss clusters similar situations together. Also receives gradients from all three downstream components during joint training.

**Key decision:** Unified dimensions — `situation_dim = text_dim = 384`. This keeps cosine fallback working for the Memory Gate and allows identity initialization of the bilinear matrix.

---

### 2. Memory Gate

Replaces cosine similarity with a trained relevance scorer.

**Architecture:** Bilinear scoring function.

```
score(situation, memory) = situation @ W @ memory.T
```

Where `W` is a (384 × 384) learned weight matrix.

**Initialization:** `W = I` (identity matrix). This means an untrained Memory Gate behaves identically to cosine similarity — smooth cold start with no performance cliff.

**Selection:** Scores all candidate memories, returns top-k with scores.

**Training:** Contrastive loss — push relevant memories' scores above irrelevant ones with a margin.

```python
loss = max(0, margin - pos_score + neg_score)  # per positive/negative pair
```

**Result:** +67% precision over cosine similarity (P@3 = 0.667 vs 0.400 on 5 real-text scenarios). Wins especially on distractor rejection and contextual relevance.

---

### 3. Strategy Selector

Learns which approach works for which situation. Two modes: **categorical** (fixed strategy set) and **continuous** (learned strategy space).

#### Categorical Mode (default)

MLP classifier over a configurable set of strategy profiles.

```
Input: situation embedding (384-dim)
Hidden: Linear(384, 128) → ReLU → Dropout
Output: Linear(128, N) → softmax → strategy probabilities
```

**Predefined strategy sets:**

| Set | Strategies | Use case |
|-----|-----------|----------|
| `generic` (7) | deep_research, quick_answer, clarify_first, step_by_step, summarize, creative, empathize | General-purpose agents |
| `developer` (12) | implement, debug, refactor, review, test, explain, architect, quick_fix, document, explore, optimize, deploy | Coding/development agents |
| `support` (7) | diagnose, guide, escalate, quick_answer, empathize, clarify, workaround | Customer support agents |

Sets are composable: `merge_strategy_sets("generic", "developer")` combines them (deduplicates by id).

Each strategy is a `StrategyProfile` with: id, name, description, prompt framing, reasoning style, response format, and tool permissions.

**Exploration:** Epsilon-greedy with temperature-scaled softmax.

**Training:** Cross-entropy loss on labeled strategy-situation pairs.

#### Continuous Mode

Instead of classifying into N categories, projects situations into a **64-dim strategy embedding space**. Strategy presets are anchor points — the model learns to blend strategies rather than picking one.

```
Input: situation embedding (384-dim)
Hidden: Linear(384, 128) → ReLU
Output: Linear(128, 64) → L2-normalize → strategy embedding
```

**Anchor points:** Learned embeddings initialized from strategy profile attributes (reasoning style, response format, tool usage). The model learns to place situations near the right anchors.

**Attribute heads:** Five continuous outputs predicted from the strategy embedding:

| Attribute | Range | Meaning |
|-----------|-------|---------|
| `reasoning_depth` | 0→1 | direct → thorough |
| `creativity` | 0→1 | conservative → exploratory |
| `verbosity` | 0→1 | concise → verbose |
| `tool_intensity` | 0→1 | no tools → heavy tool use |
| `caution` | 0→1 | confident → cautious |

**Training:** Contrastive anchor loss (pull toward correct anchor, push from others) + MSE on attribute targets.

**Why continuous?** Real conversations don't fit neat boxes. "Explain this code then refactor it" needs a blend of `explain` and `refactor`, not one or the other. The continuous space captures this naturally.

---

### 4. Confidence Estimator

Predicts whether the assembled context is sufficient for a good response.

**Architecture:** 2-layer MLP → sigmoid.

```
Input: concat(situation_emb, context_summary) → (384 + 4 = 388-dim)
Hidden: Linear(388, 64) → ReLU → Dropout
Output: Linear(64, 1) → sigmoid → confidence ∈ [0, 1]
```

**Context summary** (4-dim): top memory score, mean memory score, number of memories retrieved, strategy confidence.

**Behavioral thresholds:**
- **≥ 0.8:** Proceed normally
- **0.4 – 0.8:** Hedge — include caveats, offer to dig deeper
- **< 0.4:** Escalate — ask for clarification, state uncertainty

**Training:** Calibration loss = BCE + soft ECE (Expected Calibration Error) penalty.

```python
loss = BCE(predicted, actual) + λ * soft_ECE(predicted, actual)
```

**Result:** ECE = 0.0099 (target was < 0.1). Easy scenarios → confidence ≈ 0.99, hard scenarios → confidence ≈ 0.01.

---

## Context Assembly

The `ContextAssembler` orchestrates all four components:

```python
assembler = ContextAssembler(state_dir="./state")

result = assembler.assemble(
    query="How should I deploy?",
    memories=["We use K8s...", "Last deploy caused outage...", ...],
    k=3,
    metadata={"hour_of_day": 10},
    history=["Tests are passing"],
)

# result.selected_memories → top-k relevant memories
# result.strategy          → chosen StrategyProfile
# result.confidence        → calibrated confidence score
# result.prompt_prefix     → LLM-ready prompt with all of the above
```

The `prompt_prefix` assembles: selected memories as context, strategy framing (system prompt additions), and confidence-appropriate hedging language.

---

## Training

### Joint Training

All four components train together with shared gradients through the Situation Encoder.

```python
trainer = JointTrainer(encoder, gate, selector, estimator, registry)
metrics = trainer.train(samples, epochs=100)
```

**Multi-task loss:**
```
L_total = w₁·L_memory + w₂·L_strategy + w₃·L_confidence
```

Default weights: memory=1.0, strategy=1.0, confidence=0.5.

**Optimization:** Adam optimizer, cosine annealing LR schedule, gradient clipping (max_norm=1.0).

**Result:** Total loss 4.12 → 0.13 over 100 epochs. Memory P@2 = 0.683, strategy accuracy = 100%.

### Online Learning

After initial training, the system continues learning from real interactions via the feedback loop.

**Feedback Collector** extracts implicit training signal from user behavior:
- **Explicit:** "Thanks!" → positive, "That's wrong" → negative
- **Corrections:** "No, I meant..." → mild negative
- **Confusion:** "What do you mean?" → clarity failure
- **Engagement:** Conversation continued → mild positive, topic switched → mild negative

Pattern-based extraction with confidence weighting. Each signal maps to a reward ∈ [-1, 1].

**Replay Buffer** stores interaction outcomes (situation embedding, what was retrieved/selected, reward). Fixed capacity (default 10K), JSONL persistence, atomic writes. Recency-weighted sampling for training.

**Online Trainer** runs incremental updates:
- Small learning rate (1e-4) — stability over speed
- Updates every N interactions (default 5), not every single one
- Mini-batch sampling from replay buffer
- EMA loss tracking to detect degradation
- Gradient clipping for stability

```python
online = OnlineTrainer(encoder, gate, selector, estimator, registry)

# After each interaction:
outcome = online.record_interaction(
    situation_emb=sit, memory_indices=[0,1,2], memory_scores=[0.9,0.7,0.5],
    strategy_id="quick_answer", confidence=0.85,
    user_response="Thanks, perfect!",
)
# Auto-trains after every 5 interactions using replay buffer
```

---

## State Management

All state persists to disk. The system survives crashes, restarts, and deployments.

**StateManager** handles:
- Atomic writes (write to temp file, then rename)
- Versioned checkpoint format
- Auto-resume on startup (load latest checkpoint)
- Checkpoint pruning (keep last N)

**What's persisted:**
- Model weights for all 4 components (PyTorch `state_dict`)
- `_trained` flag as a registered buffer (survives checkpoint round-trip)
- Replay buffer (JSONL, append-only)
- Interaction logs (JSONL via `InteractionLogger`)

**Cold start:** When no checkpoint exists, Memory Gate falls back to cosine similarity (identity-initialized W), Strategy Selector explores uniformly, Confidence Estimator outputs 0.5.

---

## Benchmarks

| Benchmark | Metric | cortex-net | Baseline | Improvement |
|-----------|--------|------------|----------|-------------|
| Memory Gate (5 scenarios) | P@3 | 0.667 | 0.400 (cosine) | **+67%** |
| Situation Encoder (3 scenarios) | P@2 | 0.750 | 0.500 (no context) | **+50%** |
| Strategy Selector (10 scenarios) | Accuracy | 20%* | 10% (fixed) | **+100%** |
| Developer strategy set | Strategies | 12 | — | implement, debug, refactor, review, test, explain, architect, quick_fix, document, explore, optimize, deploy |
| Confidence Estimator | ECE | 0.010 | — | **< 0.1 target** |
| Full comparison (6 scenarios) | P@3 | 0.833 | 0.778 (cosine RAG) | **+7%** |

*Strategy selector accuracy with random embeddings (no text encoder in test); with real text embeddings, accuracy is 100% on training scenarios.

---

## File Map

```
src/cortex_net/
├── context_assembler.py      # Full pipeline: query → assembled context
├── situation_encoder.py      # MLP fusion: text + history + metadata → embedding
├── memory_gate.py            # Bilinear relevance scorer, contrastive training
├── strategy_selector.py      # 10-strategy classifier, epsilon-greedy exploration
├── confidence_estimator.py   # Calibrated confidence, BCE + ECE loss
├── joint_trainer.py          # Multi-task joint training, shared gradients
├── feedback_collector.py     # Implicit feedback extraction, replay buffer
├── online_trainer.py         # Continuous learning from interactions
├── state_manager.py          # Atomic checkpointing, auto-resume
├── interaction_log.py        # JSONL interaction logging
├── embedding_store.py        # Persistent embedding cache
├── retrieval_pipeline.py     # End-to-end retrieval pipeline
├── eval.py                   # Precision@k, recall@k metrics
├── benchmark.py              # Phase 1: memory gate real-text scenarios
├── situation_benchmark.py    # Phase 2: contextual retrieval scenarios
├── strategy_benchmark.py     # Phase 3: strategy evaluation scenarios
├── comparison_benchmark.py   # Full system vs cosine RAG vs no-memory
├── tools.py                  # Tool registry + built-in tools (file, shell)
└── agent.py                  # Live agent: cortex-net + LLM + tools
```
