"""Label conversation turns for Conversation Gate training.

Reads Atlas interaction logs, presents conversation windows, and asks
which turns are relevant to the current message. Saves labels to JSONL
for fine-tuning the gate on real data.

Usage:
    python -m cortex_net.label_conversations agents/atlas/state/conversation.json \
        --output agents/atlas/state/relevance_labels.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_conversation(path: str) -> list[dict]:
    """Load conversation history."""
    with open(path) as f:
        turns = json.load(f)
    return turns


def label_session(turns: list[dict], output_path: str, start_at: int = 0):
    """Interactive labeling session.
    
    For each turn (as "current message"), shows the preceding history
    and asks which turns are relevant.
    """
    labels = []
    
    # Resume from existing labels if any
    out = Path(output_path)
    if out.exists():
        existing = out.read_text().strip().split("\n")
        labels = [json.loads(l) for l in existing if l.strip()]
        print(f"Loaded {len(labels)} existing labels. Continuing from turn {len(labels) + 4}.")
        start_at = max(start_at, len(labels) + 4)
    
    print(f"\nLabeling conversation with {len(turns)} turns.")
    print("For each message, mark which history turns are relevant.")
    print("Enter turn numbers separated by commas, 'none' for no relevant turns, or 'q' to quit.\n")
    print("=" * 60)
    
    for i in range(max(4, start_at), len(turns)):
        current = turns[i]
        # Show up to 15 preceding turns as history
        history_start = max(0, i - 15)
        history = turns[history_start:i]
        
        print(f"\n{'=' * 60}")
        print(f"CURRENT MESSAGE (turn {i}):")
        print(f"  [{current['role']}] {current['content'][:200]}")
        print(f"\nHISTORY ({len(history)} turns):")
        
        for j, turn in enumerate(history):
            idx = history_start + j
            role_marker = "👤" if turn["role"] == "user" else "🤖"
            content = turn["content"][:120].replace("\n", " ")
            print(f"  {idx:3d} {role_marker} {content}")
        
        print(f"\nWhich history turns are relevant to turn {i}?")
        response = input(f"  Enter numbers (e.g., '{history_start},{history_start+1}'), 'none', 'all', or 'q': ").strip()
        
        if response.lower() == 'q':
            print(f"\nSaved {len(labels)} labels to {output_path}")
            break
        elif response.lower() == 'none':
            relevant = []
        elif response.lower() == 'all':
            relevant = list(range(history_start, i))
        else:
            try:
                relevant = [int(x.strip()) for x in response.split(",") if x.strip()]
                # Validate indices
                relevant = [r for r in relevant if history_start <= r < i]
            except ValueError:
                print("  Invalid input, skipping.")
                continue
        
        label = {
            "current_turn": i,
            "current_text": current["content"][:500],
            "current_role": current["role"],
            "history_start": history_start,
            "history_end": i,
            "relevant_turns": relevant,
            "total_history": len(history),
            "num_relevant": len(relevant),
        }
        labels.append(label)
        
        # Save incrementally
        with open(output_path, "w") as f:
            for l in labels:
                f.write(json.dumps(l) + "\n")
        
        print(f"  ✓ Labeled: {len(relevant)} relevant turns. ({len(labels)} total labels)")
    
    print(f"\nDone! {len(labels)} labels saved to {output_path}")
    return labels


def labels_to_training_data(
    conversation_path: str,
    labels_path: str,
) -> list[dict]:
    """Convert labels to training-ready format for the gate.
    
    Returns list of dicts with:
    - current_text: str
    - history_texts: list[str]
    - history_roles: list[str]
    - relevance: list[0|1]
    """
    turns = load_conversation(conversation_path)
    
    with open(labels_path) as f:
        labels = [json.loads(l) for l in f if l.strip()]
    
    training_data = []
    for label in labels:
        i = label["current_turn"]
        h_start = label["history_start"]
        relevant_set = set(label["relevant_turns"])
        
        history = turns[h_start:i]
        
        training_data.append({
            "current_text": turns[i]["content"],
            "history_texts": [t["content"] for t in history],
            "history_roles": [t["role"] for t in history],
            "relevance": [1 if (h_start + j) in relevant_set else 0 for j in range(len(history))],
        })
    
    return training_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m cortex_net.label_conversations <conversation.json> [--output labels.jsonl]")
        sys.exit(1)
    
    conv_path = sys.argv[1]
    output = "relevance_labels.jsonl"
    if "--output" in sys.argv:
        output = sys.argv[sys.argv.index("--output") + 1]
    
    turns = load_conversation(conv_path)
    label_session(turns, output)
