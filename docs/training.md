# Training & Learning

cortex-net has two training modes: **joint training** (batch, from labeled data) and **online learning** (continuous, from real interactions).

## Joint Training

Trains all four components together with shared gradients through the Situation Encoder.

```python
from cortex_net.joint_trainer import JointTrainer, TrainingSample

# Create training samples with labels
samples = [
    TrainingSample(
        message_emb=encoder.encode("How do I deploy?"),
        history_emb=encoder.encode("Tests are passing"),
        metadata={"hour_of_day": 10},
        memory_embs=memory_embeddings,     # (N, 384)
        relevant_indices={0, 1, 5},        # Which memories are relevant
        strategy_id="step_by_step",        # Correct strategy
        outcome=1.0,                       # 1.0 = good outcome
    ),
    ...
]

trainer = JointTrainer(encoder, gate, selector, estimator, registry)
metrics = trainer.train(samples, epochs=100)

print(metrics.summary())
# Joint Training: 100 epochs
#   Total loss:      4.12 → 0.13
#   Memory loss:     1.34 → 0.00
#   Strategy loss:   2.31 → 0.18
#   Confidence loss: 0.94 → 0.03
```

### Multi-Task Loss

```
L_total = w₁·L_memory + w₂·L_strategy + w₃·L_confidence
```

| Component | Loss | Default Weight |
|-----------|------|---------------|
| Memory Gate | Contrastive (margin-based) | 1.0 |
| Strategy Selector | Cross-entropy | 1.0 |
| Confidence Estimator | BCE + soft ECE | 0.5 |

### Optimizer

- Adam with cosine annealing LR schedule
- Gradient clipping at max_norm=1.0
- All gradients flow through the Situation Encoder

---

## Online Learning

After deployment, the system learns from real user interactions via implicit feedback.

### The Feedback Loop

```
Agent responds → User reacts → Extract signal → Update components
```

### Feedback Extraction

The `FeedbackCollector` extracts training signal from user messages:

| Signal | Examples | Reward |
|--------|----------|--------|
| **Positive** | "Thanks!", "Perfect", "That's right", 👍 | +0.5 to +1.0 |
| **Negative** | "That's wrong", "Not right", 👎 | -0.5 to -1.0 |
| **Correction** | "No, I meant...", "Actually..." | -0.2 to -0.5 |
| **Confusion** | "What?", "I don't understand", "Can you clarify?" | -0.1 to -0.3 |
| **Engagement** | Conversation continued | +0.1 |
| **Disengagement** | Topic switched | -0.1 |

Multiple signals combine. Strength depends on pattern match count.

### Experience Replay

Interactions are stored in a `ReplayBuffer` (JSONL, persistent):

```python
ReplayEntry:
    situation_emb     # What the situation looked like
    memory_indices    # Which memories were retrieved
    memory_scores     # How relevant they scored
    strategy_id       # Which strategy was selected
    confidence        # What confidence was predicted
    reward            # Aggregated feedback signal [-1, 1]
```

The buffer has a fixed capacity (default 10K) with recency-weighted sampling — recent experiences are sampled more often but old ones aren't lost.

### Online Trainer

```python
from cortex_net.online_trainer import OnlineTrainer

online = OnlineTrainer(
    encoder, gate, selector, estimator, registry,
    buffer=replay_buffer,
    lr=1e-4,            # Small LR for stability
    update_every=5,     # Train every 5 interactions
    batch_size=16,
)

# After each interaction:
outcome = online.record_interaction(
    situation_emb=situation,
    memory_indices=[0, 1, 2],
    memory_scores=[0.9, 0.7, 0.5],
    strategy_id="quick_answer",
    confidence=0.85,
    user_response="Thanks, that worked!",
)
# → positive signal extracted
# → stored in replay buffer
# → every 5 interactions, trains on a batch of 16 samples
```

### Stability Mechanisms

| Mechanism | Why |
|-----------|-----|
| Small learning rate (1e-4) | Incremental updates, not revolution |
| Experience replay | Avoids catastrophic forgetting |
| Gradient clipping (1.0) | Prevents exploding gradients |
| EMA loss tracking | Detect if learning is degrading performance |
| Update batching | Don't train on every single interaction |
| Recency-weighted sampling | Balance recent experience with historical |

### EMA Loss Tracking

The online trainer tracks exponential moving average of the training loss:

```python
online.ema_loss  # If this starts increasing, something is wrong
```

---

## Training Lifecycle

```
1. Cold start
   └─ Memory Gate uses cosine similarity (identity-initialized W)
   └─ Strategy Selector explores uniformly
   └─ Confidence Estimator outputs 0.5

2. Optional: Joint training on labeled data
   └─ Bootstrap all components from curated examples
   └─ Good for domain-specific agents

3. Live deployment with online learning
   └─ Feedback collector extracts signal from every interaction
   └─ Replay buffer accumulates experience
   └─ Components update incrementally every N interactions
   └─ Memory Gate learns which memories matter for this agent
   └─ Strategy Selector learns which approaches work
   └─ Confidence Estimator calibrates to actual outcomes

4. Periodic maintenance
   └─ MemoryBank: decay, consolidate, prune
   └─ Monitor EMA loss for degradation
   └─ Save checkpoints for rollback
```
