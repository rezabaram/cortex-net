# Monitoring

cortex-net agents log every interaction with full observability into their decisions.

## What's Logged

Each interaction records:

| Category | Fields |
|----------|--------|
| **Input** | message, length |
| **Situation** | embedding norm |
| **Memory** | count retrieved, scores, top/mean score, text previews |
| **Strategy** | selected strategy, confidence, probability distribution |
| **Confidence** | score, action (proceed/hedge/escalate) |
| **Response** | length, tool calls count, tool names |
| **Feedback** | reward (0-1), type (positive/negative/correction/neutral) |
| **Learning** | whether online update fired, EMA loss, buffer size |
| **Timing** | encoding_ms, retrieval_ms, strategy_ms, llm_ms, total_ms |

## Output Formats

### journalctl (human-readable)

One line per interaction in systemd logs:

```
[turn 3] strat=deploy conf=0.49 mem=5(top=0.07) resp=1429ch 12125ms tools=2 fb=positive(0.70)
```

View live:
```bash
journalctl --user -u atlas-slack -f | grep "turn"
```

### JSONL (machine-readable)

Full structured data in `state/<model>_interactions.jsonl`:

```json
{
  "turn_number": 3,
  "strategy_id": "deploy",
  "confidence": 0.49,
  "memories_retrieved": 5,
  "top_memory_score": 0.07,
  "memory_scores": [0.07, 0.06, 0.05, 0.04, 0.03],
  "tool_calls": 2,
  "tool_names": ["file_list", "file_read"],
  "feedback_type": "positive",
  "feedback_reward": 0.70,
  "response_length": 1429,
  "total_ms": 12125,
  "encoding_ms": 45,
  "retrieval_ms": 12,
  "llm_ms": 12050,
  "online_update": false,
  "ema_loss": 0.0,
  "buffer_size": 10
}
```

### Summary

```python
agent.monitor.summary_text()
# "📊 atlas — 42 interactions | 👍 12 👎 2 ✏️ 3 | conf=0.52 | ⏱️ 8500ms avg | 🔧 87 tool calls"
```

## What to Watch For

### Memory Gate Health
- **`top_memory_score` < 0.1**: Gate is untrained, retrieval is basically random
- **`top_memory_score` > 0.5**: Gate is learning, finding relevant memories
- **All scores similar**: Gate isn't differentiating (degenerate)

### Strategy Selection
- **Same strategy every turn**: Selector collapsed (check diversity score)
- **Strategy doesn't match task**: Untrained or needs more data
- **After training**: Strategy should match intent (debug→debug, explain→explain)

### Confidence Calibration
- **Stuck at ~0.5**: Untrained (default sigmoid output)
- **Matches actual quality**: Well-calibrated (confident on easy tasks, uncertain on hard)
- **Always high or always low**: Miscalibrated, needs more training data

### Feedback Loop
- **`buffer_size` growing**: Experiences accumulating ✅
- **`online_update: true` appearing**: Training is happening ✅
- **`ema_loss` decreasing over time**: Components are improving ✅

### Response Time
- **`encoding_ms`**: Should be <100ms (sentence-transformer)
- **`retrieval_ms`**: Should be <50ms (SQLite + gate scoring)
- **`llm_ms`**: Dominates — depends on model and tool calls
- **High `tool_calls`**: Model is exploring aggressively (cap with `max_tool_rounds`)

## Analysis

Load the JSONL for analysis:

```python
import json
import pandas as pd

with open("state/MiniMax-M2.5_interactions.jsonl") as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# Reward trend over time
df['feedback_reward'].rolling(10).mean().plot(title="Reward trend")

# Strategy distribution
df['strategy_id'].value_counts().plot(kind='bar', title="Strategy usage")

# Memory score improvement
df['top_memory_score'].rolling(10).mean().plot(title="Memory retrieval quality")

# Response time breakdown
df[['encoding_ms', 'retrieval_ms', 'llm_ms']].plot(kind='area', title="Time breakdown")
```
