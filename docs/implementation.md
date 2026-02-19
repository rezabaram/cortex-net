# Implementation Plan

## Overview

cortex-net is built in five phases, each delivering a testable component. Phase 1 (Memory Gate) is the minimum viable proof of concept — if learned memory relevance doesn't beat cosine similarity, we need to rethink the approach before investing in the rest.

---

## Phase 1: Memory Gate — Weeks 1-3

**Goal:** Prove that a learned memory scorer outperforms cosine similarity.

### Week 1: Interaction Logging
- [ ] Design interaction log schema: `(situation, memories_retrieved, memories_used, outcome_signal)`
- [ ] Build logging middleware that captures what memories were retrieved and which were actually useful
- [ ] Define outcome signals: user feedback, follow-up corrections, task completion
- [ ] Start collecting data from a live agent (cortex-memory integration)

### Week 1.5: State Management Foundation
- [ ] Implement `StateManager`: save/load checkpoints to disk with atomic writes
- [ ] Define checkpoint format: model weights + optimizer state + metadata (version, epoch, timestamp)
- [ ] Auto-resume on startup: detect latest checkpoint, load it, continue training
- [ ] Graceful cold start: if no checkpoint exists, initialize defaults (cosine similarity fallback for Memory Gate)
- [ ] State versioning: embed format version in checkpoints for forward compatibility

### Week 2: Bilinear Scorer
- [ ] Implement bilinear scoring function: `score = situation_embed · W · memory_embed`
- [ ] Build training pipeline: contrastive loss on (relevant, irrelevant) memory pairs
- [ ] Create evaluation harness: precision@k, recall@k vs. cosine baseline
- [ ] Initial training on collected interaction logs

### Week 3: Integration & Benchmarking
- [ ] Drop-in replacement for cosine similarity in cortex-memory
- [ ] A/B comparison: learned gate vs. cosine on held-out interactions
- [ ] Measure: retrieval precision, downstream task quality, latency overhead
- [ ] Document results and decision: proceed or pivot

**Exit criteria:** Memory Gate achieves measurably better retrieval precision than cosine similarity on real interaction data.

---

## Phase 2: Situation Encoder — Weeks 4-6

**Goal:** Build a shared situation representation that all components can use.

### Week 4: Feature Extraction
- [ ] Define situation features: message text, conversation history, metadata (time, user, channel, task type)
- [ ] Build feature extraction pipeline
- [ ] Tokenization and embedding strategy for mixed-type inputs

### Week 5: Encoder Architecture
- [ ] Implement 2-layer transformer encoder (or MLP baseline)
- [ ] Train on interaction logs with contrastive objective: similar situations should have similar embeddings
- [ ] Evaluate embedding quality: do similar situations cluster?

### Week 6: Integration
- [ ] Replace Memory Gate's raw input with Situation Encoder output
- [ ] Measure: does the shared encoder improve Memory Gate performance?
- [ ] Freeze or fine-tune decision based on results
- [ ] Document architecture decisions

**Exit criteria:** Situation Encoder produces meaningful embeddings where similar situations cluster together, and improves Memory Gate performance when used as input.

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
2. **Cold start:** How does the system perform before it has enough training data? Graceful degradation to cosine similarity?
3. **Catastrophic forgetting:** As the agent's domain shifts, do old learnings get overwritten?
4. **Latency:** Do the trainable components add meaningful latency to the context assembly pipeline?
5. **Transfer:** Can components trained on one agent/domain transfer to another?
6. **Self-evaluation quality:** How reliable is LLM self-rating as a training signal?
