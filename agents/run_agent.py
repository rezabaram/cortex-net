#!/usr/bin/env python3
"""Run a cortex-net agent interactively or programmatically.

Usage:
    python agents/run_agent.py atlas                    # interactive REPL
    python agents/run_agent.py atlas --query "How do I deploy?"  # single query
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cortex_net.agent import CortexAgent, AgentConfig


def load_agent(agent_name: str, api_key: str) -> CortexAgent:
    """Load an agent from its config directory."""
    agent_dir = Path(__file__).parent / agent_name
    config_path = agent_dir / "config.json"

    if not config_path.exists():
        print(f"Error: No config.json found at {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = json.load(f)

    config = AgentConfig(
        model=cfg.get("model", "MiniMax-M1"),
        base_url=cfg.get("base_url", "https://api.minimax.io/v1"),
        max_tokens=cfg.get("max_tokens", 1024),
        state_dir=str(agent_dir / "state"),
        system_prompt=cfg.get("system_prompt", "You are a helpful assistant."),
        online_learning=True,
    )

    agent = CortexAgent(config=config, api_key=api_key)

    # Seed initial memories if first run
    memories_file = agent_dir / "state" / "memories.json"
    if not memories_file.exists() and "initial_memories" in cfg:
        print(f"Seeding {len(cfg['initial_memories'])} initial memories...")
        agent.add_memories(cfg["initial_memories"])

    return agent


def interactive(agent: CortexAgent, agent_name: str):
    """Interactive REPL."""
    print(f"\n🧠 {agent_name} is ready. Type 'quit' to exit, 'stats' for stats.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "stats":
            stats = agent.stats()
            for k, v in stats.items():
                print(f"  {k}: {v}")
            continue

        response = agent.chat(user_input)
        print(f"\n{agent_name}: {response}\n")

    agent.save()
    print(f"\nState saved. {agent.stats()}")


def main():
    parser = argparse.ArgumentParser(description="Run a cortex-net agent")
    parser.add_argument("agent", help="Agent name (folder under agents/)")
    parser.add_argument("--query", "-q", help="Single query (non-interactive)")
    parser.add_argument("--api-key", "-k", help="API key", default=None)
    args = parser.parse_args()

    api_key = args.api_key or ""
    agent = load_agent(args.agent, api_key)

    if args.query:
        response = agent.chat(args.query)
        print(response)
        agent.save()
    else:
        interactive(agent, args.agent)


if __name__ == "__main__":
    main()
