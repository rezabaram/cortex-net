# Implementation Plan

## Overview

cortex-net is built in five phases, each delivering a testable component. Phase 1 (Memory Gate) is the minimum viable proof of concept — if learned memory relevance doesn't beat cosine similarity, we need to rethink the approach before investing in the rest.

---

## Phase 1: Memory Gate — ✅ COMPLETE

**Goal:** Prove that a learned memory scorer outperforms cosine similarity.

**Result: Memory Gate beats cosine by 67% on real text (0.667 vs 0.400 precision@3, 60% win rate across 5 scenarios).**

### Week 1: Interaction Logging ✅
- [x] Design interaction log schema: `(situation, memories_retrieved, memories_used, outcome_signal)`
- [x] Build logging middleware — `InteractionLogger` with append-only JSONL, fsync, corrupted-line recovery
- [x] Define outcome signals: positive, negative, neutral, unknown — with source tracking
- [x] `extract_training_pairs()` for direct Memory Gate training from logs

### Week 1.5: State Management Foundation ✅
- [x] `StateManager`: save/load checkpoints to disk with atomic writes (temp + os.replace)
- [x] Checkpoint format: model state_dict + optimizer state_dict + metadata (version, epoch, timestamp, metrics)
- [x] Auto-resume on startup: `load()` finds and loads latest checkpoint
- [x] Graceful cold start: Memory Gate falls back to cosine similarity when untrained
- [x] State versioning: format_version in every checkpoint, forward-compat check on load
- [x] Configurable checkpoint retention with auto-pruning (max_checkpoints)
- [x] `_trained` flag persisted as registered buffer (survives checkpoint round-trip)

### Week 2: Bilinear Scorer ✅
- [x] Bilinear scoring: `score = situation @ W @ memory.T` — `MemoryGate` nn.Module
- [x] Contrastive loss training on (relevant, irrelevant) memory pairs
- [x] Evaluation harness: `precision_at_k`, `recall_at_k` vs cosine baseline
- [x] Identity initialization: untrained gate ≈ cosine similarity (smooth cold start)
- [x] `EmbeddingStore`: persistent cache for real sentence-transformer embeddings

### Week 3: Integration & Benchmarking ✅
- [x] `RetrievalPipeline`: end-to-end retrieve → log → train → benchmark pipeline
- [x] Head-to-head comparison with real text via `benchmark.py`
- [x] 5 realistic scenarios: distractor traps, cross-domain relevance, temporal context, negative experience, subtle preferences
- [x] **Results documented below**

### Benchmark Results

| Scenario | Cosine P@3 | Gate P@3 | Winner |
|----------|-----------|----------|--------|
| Distractor trap | 0.33 | **0.67** | Gate |
| Cross-domain relevance | 0.67 | 0.67 | Tie |
| Temporal context | 0.67 | 0.67 | Tie |
| Negative experience | 0.33 | **0.67** | Gate |
| Subtle preference | 0.00 | **0.67** | Gate |
| **Average** | **0.400** | **0.667** | **+67%** |

Gate wins where cosine fails: distractor rejection, past failure recall, user preference detection. Loss converges from 0.86 → 0.00 in 200 epochs.

### What Was Built

| Module | File | Tests | Purpose |
|--------|------|-------|---------|
| State Manager | `state_manager.py` | 6 | Atomic checkpointing, auto-resume, pruning |
| Memory Gate | `memory_gate.py` | 8 | Bilinear scorer, contrastive training, cosine fallback |
| Interaction Logger | `interaction_log.py` | 7 | JSONL logging, training pair extraction |
| Eval Harness | `eval.py` | 6 | Metrics, synthetic data, baseline comparison |
| Embedding Store | `embedding_store.py` | 5 | Persistent embedding cache for real text |
| Benchmark | `benchmark.py` | 4 | Realistic scenarios, train + evaluate |
| Retrieval Pipeline | `retrieval_pipeline.py` | 9 | End-to-end integration |
| **Total** | **7 modules** | **45** | **All passing** |

### Known Limitations
- Training loop in `RetrievalPipeline` uses synthetic directional embeddings (not the `EmbeddingStore` — that's used in `benchmark.py`). Needs wiring together when integrated with a real memory store.
- Benchmark scenarios use curated ground truth. Real-world performance depends on quality of outcome signals from live interactions.

**Decision: Proceed to Phase 2.** ✓

---

## Phase 2: Situation Encoder — Weeks 4-6 ⏳ IN PROGRESS

**Goal:** Build a shared situation representation that all components can use.

### Week 4: Feature Extraction ✅
- [x] Define situation features: message text, conversation history, metadata (hour, day, conversation length, message length, time since last, group chat)
- [x] `extract_metadata_features()`: fixed-size numeric vector, all normalized to [0,1]
- [x] `SituationFeatures` dataclass: raw features with optional pre-computed embeddings

### Week 5: Encoder Architecture ✅
- [x] 3-layer MLP encoder (not transformer — MLP sufficient for this input structure)
- [x] Fuses: message_embedding + history_embedding + metadata → L2-normalized output
- [x] LayerNorm + Dropout for stable training
- [x] NT-Xent contrastive loss (SimCLR-style) for clustering similar situations
- [x] `encode_situation()` high-level API from raw SituationFeatures
- [x] Contrastive training test confirms: encoder learns to cluster same-type situations

### Week 6: Integration ⏳
- [x] Joint training: Situation Encoder + Memory Gate trained together on contextual scenarios
- [x] Contextual benchmark: same query, different context → different correct memories
- [ ] Wire Situation Encoder into `RetrievalPipeline` as the default path
- [ ] Measure end-to-end improvement on Phase 1 benchmark scenarios
- [ ] Document architecture decisions

### Contextual Benchmark Results

| Scenario | Encoder P@2 | Raw Text P@2 | Winner |
|----------|------------|-------------|--------|
| Status check (standup vs incident) | **0.75** | 0.50 | Encoder |
| Help request (new vs experienced) | 0.50 | 0.50 | Tie |
| Quick vs deep (DM vs group) | **1.00** | 0.50 | Encoder |
| **Average** | **0.750** | **0.500** | **+50%** |

### What Was Built

| Module | File | Tests | Purpose |
|--------|------|-------|---------|
| Situation Encoder | `situation_encoder.py` | 15 | MLP fusion, metadata extraction, contrastive loss |
| Contextual Benchmark | `situation_benchmark.py` | 2 | Context-dependent scenarios, joint training |
| **Phase 2 total** | **2 modules** | **17** | |
| **Project total** | **9 modules** | **62** | **All passing** |

**Exit criteria:** Situation Encoder produces meaningful embeddings where similar situations cluster together ✅, and improves Memory Gate performance when used as input ✅ (+50% on contextual retrieval). Remaining: wire into RetrievalPipeline.

---

## Phase 3: Strategy Selector — Weeks 7-10

**Goal:** Agent learns to choose the right approach for the right situation.

### Week 7: Strategy Profiles
- [ ] Define 10-15 initial strategy profiles (see Architecture doc)
- [ ] Each profile: prompt template + tool permissions + reasoning style + response format
- [ ] Manual labeling: classify 200+ past interactions by best strategy
- [ ] Build strategy profile registry

### Week 8-9: Classification Head
- [ ] Implement linear classifier on situation embeddings → strategy distribution
- [ ] Training pipeline: supervised on labeled interactions
- [ ] Exploration mechanism: occasionally try non-preferred strategies to learn
- [ ] Strategy outcome logging: which strategy actually led to success

### Week 10: Evaluation
- [ ] A/B test: learned strategy selection vs. fixed strategy
- [ ] Measure: task success rate, user satisfaction signals, strategy diversity
- [ ] Refine strategy profiles based on what the model actually learns
- [ ] Document results

**Exit criteria:** Learned strategy selection outperforms fixed strategy on task success metrics. The selector uses at least 5 distinct strategies regularly.

---

## Phase 4: Confidence Estimator — Weeks 11-13

**Goal:** The agent knows when it doesn't know.

### Week 11: Calibration Dataset
- [ ] Collect (situation, context, prediction, actual_outcome) tuples
- [ ] Define ground truth: binary (correct/incorrect) and graded (1-5)
- [ ] Build calibration evaluation: reliability diagrams, ECE (Expected Calibration Error)

### Week 12: Estimator Training
- [ ] Implement 2-layer MLP: situation_embed + context_summary → confidence [0, 1]
- [ ] Train with calibration loss (not just accuracy)
- [ ] Integrate behavioral thresholds: high → proceed, medium → hedge, low → escalate

### Week 13: Integration & Testing
- [ ] Wire into agent: confidence affects response framing and escalation behavior
- [ ] Measure calibration quality on held-out data
- [ ] User study: does expressed uncertainty match perceived reliability?
- [ ] Document results

**Exit criteria:** Confidence Estimator is well-calibrated (ECE < 0.1). Agent hedges on hard questions and commits on easy ones.

---

## Phase 5: Integration & End-to-End — Weeks 14-16

**Goal:** All components working together, trained end-to-end, evaluated against baseline.

### Week 14: Joint Training
- [ ] End-to-end training pipeline: all four components with shared gradients through Situation Encoder
- [ ] Loss function: weighted combination of Memory Gate relevance + Strategy success + Confidence calibration
- [ ] Training stability: gradient clipping, learning rate scheduling

### Week 15: Evaluation
- [ ] Full agent comparison: cortex-net agent vs. standard RAG agent vs. no-memory agent
- [ ] Metrics: task success, user satisfaction, learning curve (does it get better over time?), calibration
- [ ] Ablation study: which component contributes most?
- [ ] Failure analysis: where does cortex-net still fall short?

### Week 16: Documentation & Release
- [ ] Complete technical documentation
- [ ] Reproducibility: training scripts, data pipeline, evaluation harness
- [ ] Research write-up: problem, approach, results, limitations, future work
- [ ] Open source release with examples

**Exit criteria:** cortex-net agent demonstrably outperforms baseline on key metrics, with clear evidence of improvement over time (learning curve).

---

## Open Questions

These are things we don't know yet and will figure out during implementation:

1. **Data efficiency:** How many interactions before the components learn something useful? 100? 1000? 10000?
2. **Cold start:** How does the system perform before it has enough training data? Graceful degradation to cosine similarity? ← **Answered in Phase 1: yes, smooth fallback via identity-initialized W**
3. **Catastrophic forgetting:** As the agent's domain shifts, do old learnings get overwritten?
4. **Latency:** Do the trainable components add meaningful latency to the context assembly pipeline?
5. **Transfer:** Can components trained on one agent/domain transfer to another?
6. **Self-evaluation quality:** How reliable is LLM self-rating as a training signal?
