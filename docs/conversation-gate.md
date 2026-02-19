# Conversation Gate

## The Problem

Fixed conversation windows are dumb. "Last 10 messages" includes irrelevant history that confuses the LLM, and excludes relevant older messages that would help.

When you debug a NaN error for 5 messages, then ask about fibonacci, then ask "what was the NaN fix again?" — a sliding window either includes the fibonacci noise or has already dropped the debugging context.

## The Solution

A trainable gate that scores every conversation turn against the current situation and selects only what's relevant. No fixed windows, no arbitrary limits.

```
Current message → Situation Encoder → 384-dim situation embedding
                                            ↓
All history turns → Embeddings ──→ Conversation Gate ──→ Relevance scores
                                            ↓
                                  Selected turns → LLM context
```

## Architecture: Two-Tier Scoring

### Tier 1: Bilinear Scorer (Pointwise)

Scores each turn independently against the current situation.

```
score(turn_i) = σ(situation^T · W · turn_i + recency_bias - temporal_decay)
```

- **W** (384×384): Learned bilinear matrix. Identity-initialized → untrained = cosine similarity.
- **Recency bias**: Learned weight for position-based boost to recent turns.
- **Temporal decay**: Learned exponential penalty for older turns. Training found 13% penalty for oldest turn is optimal.

**147K parameters.** Handles most cases well — clear topic switches, obvious relevance, out-of-domain filtering.

### Tier 2: Cross-Attention (Contextual)

Refines pointwise scores by letting turns see each other and the situation.

```
turn_features = [turn_embedding; bilinear_score]  → project to 64-dim

Self-attention:   turns attend to each other (captures Q&A pairs, reasoning chains)
Cross-attention:  each turn attends to situation (contextual relevance)
Score head:       64-dim → 1 (refined score per turn)

final_score = (1 - gate) · bilinear_score + gate · attention_score
```

- **Residual gate**: Initialized to 0.007 (sigmoid(-5)). Untrained model = pure bilinear passthrough. Gate opens as training proves attention helps. **Zero regression risk.**
- **Self-attention**: Captures inter-turn dependencies. "If the question is selected, the answer should be too."
- **Cross-attention**: Each turn queries the situation for contextual relevance.
- **Score head**: Zero-initialized — attention contributes nothing until trained.

**74K parameters.** Currently at 0.7% activation — bilinear handles synthetic data. Will open as real conversation data reveals patterns bilinear can't capture.

### Selection

```python
# Adaptive threshold: picks the stricter of absolute vs relative
threshold = max(learned_threshold, mean_score + 0.5 * std_score)

# No artificial caps — gate decides how many turns to include
# min_turns=0: gate CAN select nothing for completely unrelated queries
selected = [turn for turn, score in zip(history, scores) if score >= threshold]
```

The relative threshold handles same-domain conversations where everything scores moderately high — it only picks turns that stand out above the crowd.

## Performance

### Embedding Cache

Full history is scored every message, but embeddings are cached incrementally — only new turns get encoded. Cost per message:

| Operation | Cost | Notes |
|-----------|------|-------|
| Embed new turns | O(1) | Only new turns, cached |
| Score all turns | O(n) | Matrix multiply, ~0.1ms for 1000 turns |
| Selection | O(n) | Threshold comparison |

Scales to 1000+ turns without meaningful latency.

### Accuracy

Tested on 4 scenarios with 8-turn mixed history (debugging + fibonacci):

| Query | Behavior | Scores |
|-------|----------|--------|
| "Back to the NaN issue" | ✅ Selected 4 debugging turns, dropped fibonacci | Debugging: 0.92-0.99, Fibonacci: 0.22-0.32 |
| "Chocolate cake recipe" | ✅ Everything near zero (nothing relevant) | All turns: 0.000-0.005 |
| "cortex-net modules?" | ✅ Near-zero, minimal selection | All turns: 0.000-0.009 |
| "Add memoization" | ✅ Selected 2 fibonacci turns | Fibonacci: 0.78-0.89, Debugging: 0.01-0.14 |

### Pre-training

Trained on 1000 synthetic conversation scenarios across 5 scenario types:

| Scenario | Description | Weight |
|----------|-------------|--------|
| `followup` | Continue same topic — all history relevant | 3x |
| `new_task` | Switch topic — nothing relevant | 2x |
| `reference_old` | Return to earlier topic — old turns relevant | 2x |
| `mixed` | Interleaved topics — only matching turns relevant | 1x |
| `subtopic_switch` | Same domain, different subtopic | 1x |
| `nothing_relevant` | Out-of-domain query — nothing relevant | 2x |

Topic clusters: 6 broad (debugging, testing, deployment, architecture, strategy, refactoring) + 8 same-domain subtopics (memory_gate_debug vs memory_gate_training, etc.) + 15 out-of-domain queries.

**Results:** Precision 0.89, Recall 0.51, Loss 0.036.

## Training

### Pre-training (Synthetic)

```bash
python -m cortex_net.pretrain_conversation_gate --save state/conversation_gate.pt
```

Generates scenarios, trains both tiers jointly with BCE loss against ground-truth relevance labels.

### Online Learning

The Conversation Gate is wired into the `OnlineTrainer` — its parameters are in the shared optimizer alongside Memory Gate, Strategy Selector, and Confidence Estimator. Feedback from real interactions flows through all components.

As the attention tier accumulates real conversation data (Q&A pairs that should stay together, multi-step debugging sessions), the residual gate will open and attention will start contributing.

## Integration

```python
# In CortexAgent.chat():
ctx = self.conversation_gate.select_turns(
    situation=situation_embedding,
    turn_embeddings=cached_history_embeddings,
    messages=full_conversation_history,
    min_turns=0,        # can select nothing
    max_turns=len(history),  # no cap
)
# ctx.messages → only relevant turns sent to LLM
```

## Files

- `src/cortex_net/conversation_gate.py` — Gate module (BilinearScorer + ContextualAttention)
- `src/cortex_net/pretrain_conversation_gate.py` — Synthetic data generation + training
- `tests/test_conversation_gate.py` — 9 tests covering scoring, selection, thresholds, training loss

## Parameters

| Component | Params | Role |
|-----------|--------|------|
| BilinearScorer.W | 147,456 | Semantic relevance matrix |
| BilinearScorer.recency_weight | 1 | Position boost |
| BilinearScorer.decay_rate | 1 | Temporal decay |
| ContextualAttention | 74,370 | Inter-turn + cross-situation attention |
| threshold_logit | 1 | Selection threshold |
| **Total** | **221,829** | |
