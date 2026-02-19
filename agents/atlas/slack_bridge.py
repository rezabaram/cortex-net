#!/usr/bin/env python3
"""Slack bridge for Atlas — polls #atlas channel and responds via cortex-net.

Uses Slack Web API only (no Socket Mode) to avoid conflicting with OpenClaw.
Runs as a persistent process (systemd service).
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import urllib.request
import urllib.parse
import urllib.error

# Add cortex-net src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cortex_net.agent import CortexAgent, AgentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("atlas-slack")

# --- Config ---
AGENT_DIR = Path(__file__).parent
CONFIG_PATH = AGENT_DIR / "config.json"

with open(CONFIG_PATH) as f:
    CFG = json.load(f)

SLACK_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
CHANNEL_ID = os.environ.get("ATLAS_CHANNEL", "C0AGR9AAAAU")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "2"))

# Bot user ID (will be fetched on startup)
BOT_USER_ID = ""


def slack_api(method: str, **kwargs) -> dict:
    """Call a Slack Web API method."""
    url = f"https://slack.com/api/{method}"
    data = urllib.parse.urlencode(kwargs).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


def get_bot_user_id() -> str:
    """Get the bot's own user ID."""
    resp = slack_api("auth.test")
    if resp.get("ok"):
        return resp["user_id"]
    raise RuntimeError(f"auth.test failed: {resp}")


def post_message(channel: str, text: str, thread_ts: str | None = None) -> None:
    """Post a message to a Slack channel."""
    kwargs = {"channel": channel, "text": text}
    if thread_ts:
        kwargs["thread_ts"] = thread_ts
    resp = slack_api("chat.postMessage", **kwargs)
    if not resp.get("ok"):
        log.error(f"Failed to post message: {resp.get('error')}")


def get_history(channel: str, oldest: str = "0", limit: int = 10) -> list[dict]:
    """Get recent messages from a channel."""
    url = f"https://slack.com/api/conversations.history?channel={channel}&oldest={oldest}&limit={limit}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {SLACK_TOKEN}"})
    resp = json.loads(urllib.request.urlopen(req).read())
    if resp.get("ok"):
        return resp.get("messages", [])
    log.error(f"conversations.history failed: {resp.get('error')}")
    return []


def create_agent() -> CortexAgent:
    """Create and seed the Atlas agent."""
    config = AgentConfig(
        model=CFG.get("model", "MiniMax-M1"),
        base_url=CFG.get("base_url", "https://api.minimax.io/v1"),
        state_dir=str(AGENT_DIR / "state"),
        system_prompt=CFG.get("system_prompt", "You are a helpful assistant."),
        online_learning=True,
        update_every=3,
        strategy_set=CFG.get("strategy_set", None),
        tools_enabled=CFG.get("tools_enabled", False),
        tools_workdir=CFG.get("tools_workdir", "."),
    )
    agent = CortexAgent(config=config, api_key=LLM_API_KEY)

    # Seed initial memories if fresh start
    if len(agent.memory_bank) == 0 and "initial_memories" in CFG:
        log.info(f"Seeding {len(CFG['initial_memories'])} initial memories")
        agent.add_memories(CFG["initial_memories"])

    return agent


def md_to_slack(text: str) -> str:
    """Convert Markdown to Slack mrkdwn format."""
    # Code blocks: ```lang\n...\n``` → ```\n...\n```  (Slack ignores lang)
    text = re.sub(r'```\w*\n', '```\n', text)

    # Headers: ### Header → *Header*
    text = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)

    # Bold: **text** → *text*
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)

    # Italic: _text_ stays the same in Slack
    # But MD single *text* for italic conflicts with Slack bold
    # Leave as-is since we converted ** to * above

    # Strikethrough: ~~text~~ → ~text~
    text = re.sub(r'~~(.+?)~~', r'~\1~', text)

    # Links: [text](url) → <url|text>
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<\2|\1>', text)

    # Inline code: `code` stays the same

    # Unordered lists: - item → • item
    text = re.sub(r'^(\s*)[-*]\s+', r'\1• ', text, flags=re.MULTILINE)

    # Ordered lists: 1. item → 1. item (Slack handles these ok)

    # Horizontal rules: --- → ───────
    text = re.sub(r'^-{3,}$', '───────────────', text, flags=re.MULTILINE)

    # Tables: convert to readable format
    lines = text.split('\n')
    result = []
    for line in lines:
        # Skip separator rows (|---|---|)
        if re.match(r'^\|[\s\-:|]+\|$', line):
            continue
        # Table rows: | a | b | → a  |  b
        if line.startswith('|') and line.endswith('|'):
            cells = [c.strip() for c in line[1:-1].split('|')]
            result.append('  '.join(cells))
        else:
            result.append(line)

    return '\n'.join(result)


def run():
    """Main loop: poll for messages, respond via Atlas."""
    global BOT_USER_ID

    if not SLACK_TOKEN:
        log.error("SLACK_BOT_TOKEN not set")
        sys.exit(1)
    if not LLM_API_KEY:
        log.error("LLM_API_KEY not set")
        sys.exit(1)

    BOT_USER_ID = get_bot_user_id()
    log.info(f"Bot user ID: {BOT_USER_ID}")

    agent = create_agent()
    log.info(f"Atlas ready. Listening on #{CHANNEL_ID}. Memories: {len(agent.memory_bank)}")

    # Persist last_ts so restarts don't skip messages
    ts_file = AGENT_DIR / "state" / "last_ts"
    ts_file.parent.mkdir(parents=True, exist_ok=True)
    if ts_file.exists():
        last_ts = ts_file.read_text().strip()
        log.info(f"Resuming from last_ts={last_ts}")
    else:
        last_ts = str(time.time() - 60)
        log.info(f"Fresh start, looking back 60s")

    while True:
        try:
            log.debug(f"Polling... last_ts={last_ts}")
            messages = get_history(CHANNEL_ID, oldest=last_ts, limit=5)
            log.debug(f"Got {len(messages)} messages")

            # Process in chronological order (API returns newest first)
            for msg in reversed(messages):
                # Skip bot's own messages
                if msg.get("user") == BOT_USER_ID:
                    continue
                if msg.get("bot_id"):
                    continue
                # Skip non-text messages
                text = msg.get("text", "").strip()
                if not text:
                    continue

                ts = msg["ts"]
                user = msg.get("user", "unknown")
                log.info(f"Message from {user}: {text[:80]}")

                # Get response from Atlas
                try:
                    response = agent.chat(text)
                    # Strip <think>...</think> tags from reasoning models
                    response = re.sub(r'<think>.*?</think>\s*', '', response, flags=re.DOTALL).strip()
                    # Convert markdown to Slack mrkdwn
                    response = md_to_slack(response)
                    post_message(CHANNEL_ID, response, thread_ts=ts)
                    log.info(f"Responded ({len(response)} chars)")
                except Exception as e:
                    log.error(f"Agent error: {e}")
                    post_message(CHANNEL_ID, f"⚠️ Error: {e}", thread_ts=ts)

                last_ts = ts
                ts_file.write_text(last_ts)

            # Save periodically
            if agent.stats()["conversation_turns"] % 20 == 0 and agent.stats()["conversation_turns"] > 0:
                agent.save()

        except KeyboardInterrupt:
            log.info("Shutting down...")
            agent.save()
            break
        except Exception as e:
            log.error(f"Poll error: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
