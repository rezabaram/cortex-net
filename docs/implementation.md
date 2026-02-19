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

## Phase 2: Situation Encoder — ✅ COMPLETE

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

### Week 6: Integration ✅
- [x] Joint training: Situation Encoder + Memory Gate trained together on contextual scenarios
- [x] Contextual benchmark: same query, different context → different correct memories
- [x] Wire Situation Encoder into `RetrievalPipeline` as optional component (`situation_dim` param)
- [x] Pipeline auto-routes: text+history+metadata → Situation Encoder → Memory Gate
- [x] Separate checkpointing for encoder and gate (both resume independently)

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

**Exit criteria:** Situation Encoder produces meaningful embeddings where similar situations cluster together ✅, and improves Memory Gate performance when used as input ✅ (+50% on contextual retrieval). Fully wired into RetrievalPipeline ✅.

**Decision: Proceed to Phase 3.** ✓

---

## Phase 3: Strategy Selector — ⏳ IN PROGRESS

**Goal:** Agent learns to choose the right approach for the right situation.

### Week 7: Strategy Profiles ✅
- [x] Defined 10 strategy profiles: deep_research, quick_answer, clarify_first, proactive_suggest, hedge_escalate, step_by_step, creative, code_assist, summarize, empathize
- [x] Each profile: prompt framing + tool permissions + reasoning style + response format
- [x] `StrategyRegistry` with lookup by id/index
- [x] `StrategyProfile` dataclass with full configuration

### Week 8-9: Classification Head ✅
- [x] 2-layer MLP classifier: situation_embedding → softmax over strategies
- [x] `train_step()` with cross-entropy loss
- [x] Epsilon-greedy exploration (configurable rate)
- [x] Temperature-scaled softmax for soft exploration
- [x] Usage tracking + diversity score (normalized entropy)
- [x] Training test: learns to map 5 distinct situations to 5 strategies (≥80% accuracy)

### Week 10: Evaluation ✅
- [x] 10 labeled scenarios covering all strategy profiles
- [x] `train_strategy_selector()`: joint encoder+selector training
- [x] Learned selection beats fixed baseline (20% vs 10% with random embs)
- [x] Uses ≥5 distinct strategies
- [x] Diversity score tracked via normalized entropy

### What Was Built

| Module | File | Tests | Purpose |
|--------|------|-------|---------|
| Strategy Selector | `strategy_selector.py` | 13 | Classification head, profiles, registry, exploration |
| **Phase 3 total** | **1 module** | **13** | |
| **Project total** | **10 modules** | **75** | **All passing** |

**Exit criteria:** Learned strategy selection outperforms fixed strategy on task success metrics. The selector uses at least 5 distinct strategies regularly.

---

## Phase 4: Confidence Estimator — ✅ COMPLETE

**Goal:** The agent knows when it doesn't know.

### Week 11-12: Estimator + Calibration ✅
- [x] `ConfidenceEstimator`: 2-layer MLP, situation_embed + context_summary → confidence [0, 1]
- [x] `ContextSummary`: retrieval quality signals (num memories, scores, spread, strategy confidence)
- [x] `CalibrationLoss`: BCE + soft ECE penalty for calibrated outputs
- [x] `expected_calibration_error()`: standard ECE metric
- [x] Behavioral thresholds: ≥0.8 proceed, 0.4-0.8 hedge, <0.4 escalate
- [x] **Training result: ECE = 0.0099** (target was < 0.1), easy conf = 0.99, hard conf = 0.01

### Week 13: Integration ✅
- [x] Full pipeline integration test: Situation Encoder → Memory Gate → Confidence Estimator
- [x] Checkpoint round-trip verified

**Exit criteria:** ECE < 0.1 ✅ (achieved 0.01). Agent hedges on hard questions and commits on easy ones ✅.

**Decision: Proceed to Phase 5.** ✓

---

## Phase 5: Integration & End-to-End — ✅ COMPLETE

**Goal:** All components working together, evaluated against baseline.

### Context Assembler ✅
- [x] `ContextAssembler`: orchestrates all four components in a single `assemble()` call
- [x] Full pipeline: query → Situation Encoder → Memory Gate + Strategy Selector + Confidence Estimator → AssembledContext
- [x] `AssembledContext.prompt_prefix`: generates LLM prompt with memories + strategy framing + confidence caveats
- [x] Save/load for all four components via StateManager
- [x] Parameter count: **1.1M total** (well under 10M target)
- [x] Unified dimensions: situation_dim = text_dim = 384 (simplifies cosine fallback, identity init)

### What Was Built (All Phases)

| Module | File | Tests | Purpose |
|--------|------|-------|---------|
| State Manager | `state_manager.py` | 6 | Atomic checkpointing, auto-resume |
| Memory Gate | `memory_gate.py` | 8 | Bilinear scorer, cosine fallback |
| Interaction Logger | `interaction_log.py` | 7 | JSONL training data logging |
| Eval Harness | `eval.py` | 6 | Metrics + synthetic data |
| Embedding Store | `embedding_store.py` | 5 | Persistent embedding cache |
| Benchmark | `benchmark.py` | 4 | Phase 1 real-text scenarios |
| Retrieval Pipeline | `retrieval_pipeline.py` | 9 | End-to-end retrieval |
| Situation Encoder | `situation_encoder.py` | 15 | MLP fusion, metadata-aware |
| Situation Benchmark | `situation_benchmark.py` | 2 | Contextual scenarios |
| Strategy Selector | `strategy_selector.py` | 13 | Classification head, 10 profiles |
| Strategy Benchmark | `strategy_benchmark.py` | 2 | Labeled scenarios, eval vs fixed |
| Confidence Estimator | `confidence_estimator.py` | 13 | Calibrated confidence, ECE |
| Context Assembler | `context_assembler.py` | 6 | Full pipeline orchestration |
| Joint Trainer | `joint_trainer.py` | 4 | Multi-task training, ablation |
| Comparison Benchmark | `comparison_benchmark.py` | 2 | cortex-net vs RAG vs none |
| Feedback Collector | `feedback_collector.py` | 15 | Implicit signal extraction, replay buffer |
| Online Trainer | `online_trainer.py` | 7 | Continuous learning from interactions |
| Memory Bank | `memory_bank.py` | 19 | SQLite-backed extensible memory |
| Tools | `tools.py` | 17 | File, shell, edit tools + registry |
| Agent | `agent.py` | 10 | Live agent with LLM + tools + learning |
| **Total** | **20 modules** | **170** | **All passing** |

### Joint Training ✅
- [x] `JointTrainer`: single optimizer across all components with shared gradients through Situation Encoder
- [x] Weighted multi-task loss: memory (contrastive) + strategy (cross-entropy) + confidence (calibration)
- [x] Gradient clipping + cosine annealing LR schedule
- [x] **Results**: total loss 4.12 → 0.13, memory P@2 = 0.683, strategy acc = 100%
- [x] Ablation framework with per-component evaluation

### Comparison Benchmark ✅
- [x] 6 realistic scenarios with real text embeddings (sentence-transformers)
- [x] cortex-net: P@3=0.833, R@3=0.875, strategy=100%, confidence=100%
- [x] cosine RAG: P@3=0.778, R@3=0.792
- [x] **+7% precision** over standard RAG, plus strategy + confidence capabilities

### Documentation & Release ✅
- [x] README with results table, architecture diagram, quick start, full API
- [x] Proper pyproject.toml with hatchling build
- [x] mkdocs served via systemd (auto-restart)

### Feedback Collector + Online Learning ✅
- [x] Pattern-based feedback extraction (positive/negative/correction/confusion)
- [x] Experience replay buffer (JSONL, persistent, recency-weighted sampling)
- [x] Online trainer: incremental updates every N interactions, EMA loss tracking
- [x] Small LR (1e-4) + gradient clipping for stability

### MemoryBank ✅
- [x] SQLite-backed with WAL mode, binary-packed embeddings
- [x] Content types: text, image, file, audio, video, structured
- [x] Metadata: source, importance, access count, timestamps, tags, parent_id
- [x] Maintenance: importance decay, consolidation, pruning
- [x] Replaces flat JSON memory store

### Live Agent ✅
- [x] `CortexAgent`: full pipeline (context assembly → LLM → feedback → learning)
- [x] OpenAI-compatible API (tested with MiniMax M1)
- [x] Auto-memorization of important exchanges
- [x] State persistence (weights, memories, conversation, replay buffer)

### Tool System ✅
- [x] `ToolRegistry` with OpenAI function calling schema generation
- [x] Built-in tools: `file_read`, `file_write`, `file_edit`, `file_list`, `shell`
- [x] Tool loop: LLM calls tools iteratively (max 10 rounds)
- [x] Safety: timeouts, output caps, dangerous flag

### Slack Integration ✅
- [x] Polling-based Slack bridge (systemd service, auto-restart)
- [x] Atlas agent live in `#atlas` channel with full tool access

### Remaining
- [ ] **Learned Memorization** — classifier that decides what to store and at what importance (trained from feedback signals). Critical at 1,000+ memories.
- [ ] **Memory Extraction** — auto-summarize long exchanges into concise memories
- [ ] **Adaptive Consolidation** — learn optimal similarity thresholds and trigger timing
- [ ] Extended ablation with real text encodings
- [ ] Long-running validation (does online learning improve over 100s of interactions?)

**Exit criteria:** cortex-net agent demonstrably outperforms baseline on key metrics, with clear evidence of improvement over time (learning curve).

---

---

## Phase 6: Scale Benchmark — 🔜 NEXT

**Goal:** Prove cortex-net's context assembly outperforms baselines at scale, where "just shove everything in the context window" stops working.

### Why this matters
With 1M token context windows, naive approaches work fine for small sessions. cortex-net's value is selecting the right 0.1% from a massive history. This phase proves that claim with numbers.

### Synthetic History Generator
- [ ] Generate 6 months of realistic agent interaction history
- [ ] 10-20K conversation turns across diverse topics (coding, debugging, architecture, ops, preferences)
- [ ] Temporal structure: recurring themes, topic drift, seasonal patterns
- [ ] Embed everything, load into MemoryBank

### Evaluation Dataset
- [ ] 50 queries requiring specific buried knowledge (not surface-level text matches)
- [ ] Categories: cross-session recall, preference memory, failure pattern recognition, temporal context ("what did we decide last month about X?")
- [ ] Ground truth: which memories/turns should be retrieved for each query

### Baselines
- [ ] **Cosine top-k**: standard RAG with sentence-transformer similarity
- [ ] **BM25**: keyword-based retrieval
- [ ] **Random sample**: control baseline
- [ ] **Full context**: everything in the window (up to token limit) — tests whether LLM intelligence alone suffices
- [ ] **cortex-net**: Memory Gate + Situation Encoder + Conversation Gate

### Scoring Pipeline
- [ ] Automated: Precision@k, Recall@k, MRR
- [ ] LLM-as-judge: response quality scoring on a 1-5 scale
- [ ] Cost metric: tokens used per query (cortex-net should use 10-100x fewer)
- [ ] Latency: end-to-end response time

**Exit criteria:** cortex-net beats cosine top-k by ≥15% on precision and uses ≥10x fewer tokens than full-context approach.

---

## Phase 7: Integration Layer — 🔜 PLANNED

**Goal:** Package cortex-net as a reusable context preprocessing layer that any LLM application can use.

### Core API
- [ ] `ContextFilter`: input = (query, history, memories) → output = optimally selected subset
- [ ] Provider-agnostic: works with OpenAI, Anthropic, local models
- [ ] Token-budget-aware: "give me the best context that fits in N tokens"
- [ ] Pluggable into OpenClaw, LangChain, raw API calls

### Deployment modes
- [ ] Library: `pip install cortex-net`, import and use
- [ ] Service: HTTP endpoint for context filtering
- [ ] OpenClaw plugin: native integration (already partially done via cortex-memory extension)

### Pre-trained weights
- [ ] Ship default weights trained on Phase 6 synthetic data
- [ ] Fine-tuning API for domain-specific adaptation
- [ ] Zero-shot mode: works without training (cosine fallback)

**Exit criteria:** A developer can `pip install cortex-net` and get better context assembly in <10 lines of code.

---

## Open Questions

These are things we don't know yet and will figure out during implementation:

1. **Data efficiency:** How many interactions before the components learn something useful? 100? 1000? 10000?
2. **Cold start:** How does the system perform before it has enough training data? Graceful degradation to cosine similarity? ← **Answered in Phase 1: yes, smooth fallback via identity-initialized W**
3. **Catastrophic forgetting:** As the agent's domain shifts, do old learnings get overwritten?
4. **Latency:** Do the trainable components add meaningful latency to the context assembly pipeline?
5. **Transfer:** Can components trained on one agent/domain transfer to another?
6. **Self-evaluation quality:** How reliable is LLM self-rating as a training signal?
