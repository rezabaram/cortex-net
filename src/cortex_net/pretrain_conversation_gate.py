"""Pre-train Conversation Gate on synthetic conversation scenarios.

Generates realistic multi-topic conversation histories where we KNOW
which turns are relevant to the current message. Trains the gate to
distinguish relevant context from irrelevant noise.

Scenarios:
1. New task after long discussion → only new task relevant
2. Follow-up question → recent related turns relevant
3. Reference to earlier topic → earlier turns relevant, not recent ones
4. Multi-topic conversation → only turns matching current topic
"""

from __future__ import annotations

import random
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from cortex_net.conversation_gate import ConversationGate

# Synthetic conversation topics — each is a cluster of related messages
# Broad topics for cross-domain distinction
TOPICS = {
    "debugging": [
        "There's a bug in the memory gate — it returns NaN scores.",
        "The NaN comes from the bilinear matrix. Try clamping the output.",
        "That fixed the NaN but now all scores are near zero.",
        "Check if the embeddings are normalized before scoring.",
        "Normalizing fixed it. Scores look reasonable now.",
        "The scoring still seems off for long documents.",
        "Long documents get penalized because of L2 normalization.",
        "We should try a different normalization strategy for long inputs.",
    ],
    "testing": [
        "We need tests for the monitor module.",
        "Write tests covering JSONL output and summary stats.",
        "The test should use tmp_path fixture for isolation.",
        "Make sure to test edge cases — empty logs, missing fields.",
        "All 9 monitor tests pass. Good coverage.",
        "We should add integration tests for the full pipeline.",
        "The agent tests need mocking for the LLM calls.",
        "Run pytest with coverage to find untested code paths.",
    ],
    "deployment": [
        "How do we deploy the agent to production?",
        "Use systemd for process management.",
        "The service file should set WorkingDirectory correctly.",
        "We need health checks and automatic restarts.",
        "Set up log rotation for the interaction logs.",
        "Docker would be better for reproducible deployments.",
        "The memory database needs to be persisted across restarts.",
        "Configure WAL mode for SQLite in production.",
    ],
    "architecture": [
        "Explain the Situation Encoder architecture.",
        "It's a 2-layer MLP fusing text, history, and metadata to 384-dim.",
        "Why MLP instead of a transformer?",
        "At our scale with limited data, MLP is sufficient and trains faster.",
        "The unified 384 dimensions enable cosine fallback.",
        "Should we add attention over memory items?",
        "Not yet — the bottleneck is training data, not model capacity.",
        "We could add cross-attention between situation and memories later.",
    ],
    "strategy": [
        "The strategy selector keeps picking 'optimize' for everything.",
        "That's because it's untrained — random selection from 12 strategies.",
        "How do we train it to pick better strategies?",
        "It learns from feedback — positive feedback reinforces the chosen strategy.",
        "The developer strategy set has implement, debug, refactor, review, test.",
        "We should add a 'planning' strategy for architectural discussions.",
        "The continuous strategy selector blends strategies with weights.",
        "Anchor points in strategy space define each strategy's attributes.",
    ],
    "refactoring": [
        "The slack bridge is getting too complex. Let's refactor.",
        "Extract the message formatting into its own module.",
        "The polling loop should be separate from message handling.",
        "We should add type hints to all public functions.",
        "Split the monolithic bridge into smaller, testable components.",
        "The markdown-to-slack conversion should be its own function.",
        "Consider an event-driven architecture instead of polling.",
        "Use dataclasses for message types instead of raw dicts.",
    ],
}


# Same-domain subtopics — harder to distinguish
SUBTOPICS = {
    "memory_gate_debug": [
        "The memory gate is returning NaN scores on long inputs.",
        "I think the bilinear matrix is overflowing. Let's clamp.",
        "After clamping, scores are all near zero for some queries.",
        "The embeddings aren't normalized — that explains the zero scores.",
        "Gate scores look good now. Identity init works as cosine fallback.",
    ],
    "memory_gate_training": [
        "How do we train the memory gate? Contrastive loss?",
        "Yes, positive pairs from same conversation, negatives from others.",
        "The training converges fast — 10 epochs is enough for memory gate.",
        "We should use hard negatives for better precision.",
        "Gate precision went from 0.40 to 0.67 after training.",
    ],
    "strategy_debugging": [
        "The strategy selector keeps picking 'optimize' for every query.",
        "That's because it's untrained — epsilon-greedy gives random picks.",
        "Even after training, it picks 'deploy' for code review questions.",
        "We need more training data for strategy to differentiate tasks.",
        "The categorical selector can't blend strategies. That's a limitation.",
    ],
    "strategy_design": [
        "Should we use continuous strategies instead of categorical?",
        "Yes — anchor points in 64-dim space with attribute heads.",
        "The five attributes are reasoning_depth, creativity, verbosity, tool_intensity, caution.",
        "Strategy blending lets us combine implement and debug approaches.",
        "ContinuousStrategySelector projects situation to strategy space.",
    ],
    "testing_monitor": [
        "We need tests for the monitor module.",
        "Cover JSONL output, summary stats, and resume from existing logs.",
        "Use tmp_path fixture for test isolation.",
        "The InteractionLog dataclass needs default values for all fields.",
        "All 9 monitor tests pass. Good coverage.",
    ],
    "testing_agent": [
        "The agent tests need mocking for LLM calls.",
        "We mock OpenAI client to return canned responses.",
        "Test the feedback loop: positive message should yield high reward.",
        "Agent state save/load needs round-trip testing.",
        "The tool loop test should verify max_tool_rounds is respected.",
    ],
    "deployment_systemd": [
        "Set up atlas-slack.service as a systemd user service.",
        "WorkingDirectory should be the cortex-net repo root.",
        "Enable the service so it starts on boot.",
        "Log rotation: systemd journal handles it but we also write JSONL.",
        "Restart=always with 5-second delay for crash recovery.",
    ],
    "deployment_docker": [
        "We should containerize the agent with Docker.",
        "The Dockerfile needs sentence-transformers and PyTorch.",
        "Mount the state directory as a volume for persistence.",
        "Use multi-stage build to keep the image small.",
        "Docker compose for the agent plus a monitoring sidecar.",
    ],
}


# Out-of-domain messages — things with NO relation to any topic above
OUT_OF_DOMAIN = [
    "What's a good recipe for chocolate cake?",
    "Can you recommend a movie for tonight?",
    "What's the capital of Mongolia?",
    "Tell me a joke about penguins.",
    "How do I fix a leaky faucet?",
    "What's the meaning of life?",
    "Write me a haiku about rain.",
    "Who won the World Cup in 2022?",
    "What's the best way to learn piano?",
    "How far is the moon from Earth?",
    "Recommend a good book about history.",
    "What's the weather like in Tokyo?",
    "How do you make sourdough bread?",
    "Explain quantum entanglement simply.",
    "What's a good workout routine for beginners?",
]


def generate_scenario() -> tuple[str, list[str], list[int]]:
    """Generate a conversation scenario with relevance labels.
    
    Returns:
        current_message: the current user message
        history: list of conversation turns
        relevant_indices: which history turns are relevant
    """
    # Mix broad topics and subtopics for training diversity
    all_pools = {**TOPICS, **SUBTOPICS}
    topics = list(all_pools.keys())
    scenario_type = random.choice([
        "new_task", "followup", "reference_old", "mixed",
        "subtopic_switch",
        "nothing_relevant", "nothing_relevant", "nothing_relevant",  # 3x weight — hardest case
    ])
    
    if scenario_type == "new_task":
        # Long conversation about topic A, then new task about topic B
        topic_a, topic_b = random.sample(topics, 2)
        history = random.sample(all_pools[topic_a], min(6, len(all_pools[topic_a])))
        current_message = random.choice(all_pools[topic_b])
        relevant_indices = []
        
    elif scenario_type == "followup":
        # Conversation about topic A, follow-up about topic A
        topic = random.choice(topics)
        msgs = all_pools[topic]
        n_hist = random.randint(3, min(6, len(msgs) - 1))
        history = msgs[:n_hist]
        current_message = msgs[n_hist]
        relevant_indices = list(range(len(history)))
        
    elif scenario_type == "reference_old":
        # Topic A → Topic B → back to Topic A
        topic_a, topic_b = random.sample(topics, 2)
        hist_a = random.sample(all_pools[topic_a], 3)
        hist_b = random.sample(all_pools[topic_b], 3)
        history = hist_a + hist_b
        current_message = random.choice([m for m in all_pools[topic_a] if m not in hist_a])
        relevant_indices = [0, 1, 2]
        
    elif scenario_type == "mixed":
        # Interleaved topics, ask about one
        topic_a, topic_b = random.sample(topics, 2)
        msgs_a = random.sample(all_pools[topic_a], 3)
        msgs_b = random.sample(all_pools[topic_b], 3)
        history = []
        indices_a = []
        for i in range(3):
            history.append(msgs_a[i])
            indices_a.append(len(history) - 1)
            history.append(msgs_b[i])
        current_message = random.choice([m for m in all_pools[topic_a] if m not in msgs_a])
        relevant_indices = indices_a
    
    elif scenario_type == "subtopic_switch":
        # Same broad domain but different subtopic
        subtopic_keys = list(SUBTOPICS.keys())
        domains = {}
        for k in subtopic_keys:
            prefix = k.rsplit("_", 1)[0]
            domains.setdefault(prefix, []).append(k)
        multi_domains = {k: v for k, v in domains.items() if len(v) >= 2}
        if multi_domains:
            domain = random.choice(list(multi_domains.keys()))
            st_a, st_b = random.sample(multi_domains[domain], 2)
        else:
            st_a, st_b = random.sample(subtopic_keys, 2)
        history = random.sample(SUBTOPICS[st_a], min(4, len(SUBTOPICS[st_a])))
        current_message = random.choice(SUBTOPICS[st_b])
        relevant_indices = []
    
    elif scenario_type == "nothing_relevant":
        # Out-of-domain query after technical conversation — NOTHING is relevant
        # Mix multiple technical topics in history for realism
        n_topics = random.randint(1, 3)
        selected_topics = random.sample(topics, min(n_topics, len(topics)))
        history = []
        for t in selected_topics:
            history.extend(random.sample(all_pools[t], min(3, len(all_pools[t]))))
        random.shuffle(history)
        history = history[:random.randint(4, 8)]
        current_message = random.choice(OUT_OF_DOMAIN)
        relevant_indices = []  # nothing in history is relevant
    
    return current_message, history, relevant_indices


def pretrain(
    gate: ConversationGate,
    encoder: SentenceTransformer,
    n_scenarios: int = 500,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """Pre-train the Conversation Gate on synthetic scenarios.
    
    Returns dict with training stats.
    """
    gate = gate.to(device)
    gate.train()
    optimizer = torch.optim.Adam(gate.parameters(), lr=lr)
    
    # Generate all scenarios and embed them
    if verbose:
        print(f"Generating {n_scenarios} synthetic scenarios...")
    
    scenarios = []
    for _ in range(n_scenarios):
        current, history, relevant = generate_scenario()
        if not history:
            continue
            
        # Embed
        all_texts = [current] + history
        embs = encoder.encode(all_texts, convert_to_tensor=True).to(device).clone()
        current_emb = embs[0]
        history_embs = embs[1:]
        
        # Create labels
        labels = torch.zeros(len(history), device=device)
        for idx in relevant:
            if idx < len(history):
                labels[idx] = 1.0
        
        scenarios.append((current_emb, history_embs, labels))
    
    if verbose:
        print(f"Generated {len(scenarios)} valid scenarios. Training...")
    
    # Train
    losses = []
    for epoch in range(epochs):
        random.shuffle(scenarios)
        epoch_loss = 0.0
        
        for current_emb, history_embs, labels in scenarios:
            optimizer.zero_grad()
            loss = gate.training_loss(current_emb, history_embs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(scenarios)
        losses.append(avg_loss)
        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
    
    gate._trained.fill_(True)
    gate.eval()
    
    return {
        "scenarios": len(scenarios),
        "epochs": epochs,
        "final_loss": losses[-1],
        "loss_reduction": f"{losses[0]:.4f} → {losses[-1]:.4f}",
    }


def evaluate(
    gate: ConversationGate,
    encoder: SentenceTransformer,
    n_scenarios: int = 100,
    device: str = "cpu",
) -> dict:
    """Evaluate the gate on held-out scenarios."""
    gate.eval()
    
    correct_selections = 0
    total_scenarios = 0
    precision_sum = 0.0
    recall_sum = 0.0
    
    for _ in range(n_scenarios):
        current, history, relevant = generate_scenario()
        if not history:
            continue
        
        all_texts = [current] + history
        embs = encoder.encode(all_texts, convert_to_tensor=True).to(device)
        current_emb = embs[0]
        history_embs = embs[1:]
        
        with torch.no_grad():
            selected_mask, scores = gate.forward(
                current_emb, history_embs, min_turns=1, max_select=len(history)
            )
        
        selected_set = set(selected_mask.nonzero(as_tuple=True)[0].tolist())
        relevant_set = set(relevant)
        
        if relevant_set:
            # Recall: what fraction of relevant turns were selected?
            recall = len(selected_set & relevant_set) / len(relevant_set)
            recall_sum += recall
        
        if selected_set:
            # Precision: what fraction of selected turns were relevant?
            if relevant_set:
                precision = len(selected_set & relevant_set) / len(selected_set)
            else:
                # No relevant turns — selecting fewer is better
                precision = 1.0 - len(selected_set) / len(history)
            precision_sum += precision
        
        total_scenarios += 1
    
    return {
        "scenarios": total_scenarios,
        "avg_precision": round(precision_sum / total_scenarios, 3),
        "avg_recall": round(recall_sum / total_scenarios, 3),
    }


if __name__ == "__main__":
    import sys
    
    device = "cpu"
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Baseline (untrained)
    gate = ConversationGate(dim=384).to(device)
    print("=== Baseline (untrained) ===")
    baseline = evaluate(gate, encoder, n_scenarios=200, device=device)
    print(f"  Precision: {baseline['avg_precision']}")
    print(f"  Recall: {baseline['avg_recall']}")
    print()
    
    # Pre-train
    print("=== Pre-training ===")
    stats = pretrain(gate, encoder, n_scenarios=1000, epochs=20, device=device)
    print(f"\n  {stats}")
    print()
    
    # After training
    print("=== After Pre-training ===")
    after = evaluate(gate, encoder, n_scenarios=200, device=device)
    print(f"  Precision: {after['avg_precision']}")
    print(f"  Recall: {after['avg_recall']}")
    print()
    
    # Save if requested
    if "--save" in sys.argv:
        path = sys.argv[sys.argv.index("--save") + 1] if len(sys.argv) > sys.argv.index("--save") + 1 else "conversation_gate.pt"
        torch.save(gate.state_dict(), path)
        print(f"Saved to {path}")
