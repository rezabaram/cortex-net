#!/usr/bin/env python3
"""Slack bridge for Atlas — polls a channel, responds via cortex-net.

Uses Slack Web API (no Socket Mode) to coexist with OpenClaw.
Runs as a systemd user service.
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from cortex_net.agent import CortexAgent, AgentConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("atlas-slack")

AGENT_DIR = Path(__file__).parent
STATE_DIR = AGENT_DIR / "state"
TS_FILE = STATE_DIR / "last_ts"

# ── Slack helpers ──────────────────────────────────────────────────────────────

class SlackClient:
    """Minimal Slack Web API client."""

    def __init__(self, token: str, channel: str):
        self.token = token
        self.channel = channel
        self.bot_user_id = self._call("auth.test")["user_id"]

    def _call(self, method: str, **kwargs) -> dict:
        data = urlencode(kwargs).encode()
        req = Request(
            f"https://slack.com/api/{method}",
            data=data,
            headers={"Authorization": f"Bearer {self.token}"},
        )
        resp = json.loads(urlopen(req).read())
        if not resp.get("ok"):
            raise RuntimeError(f"Slack {method}: {resp.get('error', resp)}")
        return resp

    def history(self, oldest: str, limit: int = 5) -> list[dict]:
        return self._call("conversations.history", channel=self.channel, oldest=oldest, limit=limit).get("messages", [])

    def post(self, text: str) -> None:
        self._call("chat.postMessage", channel=self.channel, text=text)


# ── Markdown → Slack mrkdwn ───────────────────────────────────────────────────

def md_to_slack(text: str) -> str:
    text = re.sub(r"```\w*\n", "```\n", text)                              # strip lang from fences
    text = re.sub(r"^#{1,6}\s+(.+)$", r"*\1*", text, flags=re.MULTILINE)   # headers → bold
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)                         # **bold** → *bold*
    text = re.sub(r"~~(.+?)~~", r"~\1~", text)                             # ~~strike~~ → ~strike~
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)            # [t](u) → <u|t>
    text = re.sub(r"^(\s*)[-*]\s+", r"\1• ", text, flags=re.MULTILINE)     # bullets → •
    text = re.sub(r"^-{3,}$", "───────", text, flags=re.MULTILINE)         # hr
    # tables: drop separator rows, strip outer pipes
    lines = []
    for line in text.split("\n"):
        if re.match(r"^\|[\s\-:|]+\|$", line):
            continue
        if line.startswith("|") and line.endswith("|"):
            line = "  ".join(c.strip() for c in line[1:-1].split("|"))
        lines.append(line)
    return "\n".join(lines)


# ── Agent factory ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(AGENT_DIR / "config.json") as f:
        return json.load(f)


def create_agent(cfg: dict) -> CortexAgent:
    backend = cfg.get("backend", "openai")
    oauth_token = None
    api_key: str | None = os.environ.get("LLM_API_KEY")

    if backend == "anthropic":
        auth_path = Path.home() / ".openclaw/agents/main/agent/auth-profiles.json"
        if auth_path.exists():
            with open(auth_path) as f:
                oauth_token = json.load(f).get("profiles", {}).get("anthropic:default", {}).get("token")
        api_key = None

    config = AgentConfig(
        model=cfg.get("model", "MiniMax-M1"),
        base_url=cfg.get("base_url", "https://api.minimax.io/v1"),
        backend=backend,
        anthropic_oauth_token=oauth_token,
        state_dir=str(STATE_DIR),
        system_prompt=cfg.get("system_prompt", "You are a helpful assistant."),
        online_learning=True,
        update_every=3,
        strategy_set=cfg.get("strategy_set"),
        tools_enabled=cfg.get("tools_enabled", False),
        tools_workdir=cfg.get("tools_workdir", "."),
        max_tool_rounds=cfg.get("max_tool_rounds", 25),
        chat_timeout=cfg.get("chat_timeout", 120),
    )
    agent = CortexAgent(config=config, api_key=api_key)

    if len(agent.memory_bank) == 0 and "initial_memories" in cfg:
        log.info(f"Seeding {len(cfg['initial_memories'])} memories")
        agent.add_memories(cfg["initial_memories"])

    return agent


# ── Main loop ─────────────────────────────────────────────────────────────────

MAX_CONSECUTIVE_ERRORS = 5
BACKOFF_BASE = 5          # seconds
BACKOFF_MAX = 300         # 5 min cap

def run():
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    channel = os.environ.get("ATLAS_CHANNEL", "C0AGR9AAAAU")
    poll_interval = float(os.environ.get("POLL_INTERVAL", "2"))

    if not token:
        log.error("SLACK_BOT_TOKEN not set")
        sys.exit(1)

    cfg = load_config()
    if cfg.get("backend", "openai") != "anthropic" and not os.environ.get("LLM_API_KEY"):
        log.error("LLM_API_KEY not set")
        sys.exit(1)

    slack = SlackClient(token, channel)
    log.info(f"Bot user: {slack.bot_user_id}")

    agent = create_agent(cfg)
    log.info(f"Atlas ready — {len(agent.memory_bank)} memories, model={cfg.get('model')}")

    # Resume from last processed timestamp
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    last_ts = TS_FILE.read_text().strip() if TS_FILE.exists() else str(time.time())

    consecutive_errors = 0

    while True:
        try:
            messages = slack.history(oldest=last_ts)

            # Chronological order (API returns newest-first)
            for msg in reversed(messages):
                ts = msg["ts"]
                user = msg.get("user", "")
                text = msg.get("text", "").strip()

                # Skip: own messages, empty, subtypes (joins, pins, etc.)
                if user == slack.bot_user_id or not text or msg.get("subtype"):
                    last_ts = ts
                    continue

                log.info(f"[{user}] {text[:100]}")

                try:
                    response = agent.chat(text)
                    # Strip <think> tags from reasoning models
                    response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
                    if not response:
                        response = "(no response generated)"
                    slack.post(md_to_slack(response))
                    log.info(f"Replied ({len(response)} chars)")
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    log.error(f"LLM error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}")

                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        backoff = min(BACKOFF_BASE * (2 ** consecutive_errors), BACKOFF_MAX)
                        log.warning(f"Too many errors, backing off {backoff}s")
                        time.sleep(backoff)

                last_ts = ts

            # Persist cursor
            TS_FILE.write_text(last_ts)

            # Save agent state every 20 turns
            turns = agent.stats()["conversation_turns"]
            if turns > 0 and turns % 20 == 0:
                agent.save()

        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"Poll error: {e}")

        time.sleep(poll_interval)

    agent.save()
    log.info("Shutdown complete")


if __name__ == "__main__":
    run()
