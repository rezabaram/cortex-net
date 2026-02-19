"""Web-based labeling tool for Conversation Gate training.

Run: python -m cortex_net.label_web agents/atlas/state/conversation.json
Open: http://localhost:8501
"""

from __future__ import annotations

import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs

CONV_PATH = ""
OUTPUT_PATH = ""
TURNS: list[dict] = []
LABELS: list[dict] = []
CURRENT_IDX = 4


def load_state():
    global TURNS, LABELS, CURRENT_IDX
    with open(CONV_PATH) as f:
        TURNS = json.load(f)
    
    out = Path(OUTPUT_PATH)
    if out.exists():
        LABELS = [json.loads(l) for l in out.read_text().strip().split("\n") if l.strip()]
        CURRENT_IDX = max(4, len(LABELS) + 4)
    else:
        LABELS = []
        CURRENT_IDX = 4


def save_labels():
    with open(OUTPUT_PATH, "w") as f:
        for l in LABELS:
            f.write(json.dumps(l) + "\n")


def render_page() -> str:
    global CURRENT_IDX
    
    if CURRENT_IDX >= len(TURNS):
        return "<html><body><h1>✅ All turns labeled!</h1><p>{} labels saved.</p></body></html>".format(len(LABELS))
    
    current = TURNS[CURRENT_IDX]
    history_start = max(0, CURRENT_IDX - 15)
    history = TURNS[history_start:CURRENT_IDX]
    
    history_html = ""
    for j, turn in enumerate(history):
        idx = history_start + j
        role_icon = "👤" if turn["role"] == "user" else "🤖"
        content = turn["content"][:300].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        history_html += f"""
        <div style="margin:8px 0; padding:8px; background:{'#1a1a2e' if turn['role']=='user' else '#16213e'}; border-radius:8px; display:flex; align-items:flex-start; gap:8px;">
            <label style="white-space:nowrap; cursor:pointer;">
                <input type="checkbox" name="relevant" value="{idx}" style="transform:scale(1.3); margin-top:4px;">
                <span style="font-weight:bold; color:#aaa;">{idx}</span> {role_icon}
            </label>
            <div style="color:#ddd; font-size:14px;">{content}</div>
        </div>"""
    
    current_content = current["content"][:500].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    current_icon = "👤" if current["role"] == "user" else "🤖"
    
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Label Conversations — cortex-net</title>
    <style>
        body {{ background: #0a0a1a; color: #eee; font-family: -apple-system, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .current {{ background: #1e3a5f; padding: 16px; border-radius: 12px; border: 2px solid #4a9eff; margin: 20px 0; }}
        .btn {{ padding: 10px 24px; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; margin: 4px; }}
        .btn-primary {{ background: #4a9eff; color: white; }}
        .btn-secondary {{ background: #333; color: #aaa; }}
        .btn:hover {{ opacity: 0.9; }}
        .progress {{ color: #888; font-size: 14px; }}
    </style>
</head>
<body>
    <h2>🕸️ Conversation Gate — Label Training Data</h2>
    <p class="progress">Turn {CURRENT_IDX}/{len(TURNS)} · {len(LABELS)} labels saved</p>
    
    <div class="current">
        <strong>CURRENT MESSAGE (turn {CURRENT_IDX}):</strong><br>
        {current_icon} <span style="color:#4a9eff;">[{current['role']}]</span> {current_content}
    </div>
    
    <p><strong>Which history turns are relevant?</strong> Check the boxes:</p>
    
    <form method="POST" action="/label">
        <input type="hidden" name="turn_idx" value="{CURRENT_IDX}">
        {history_html}
        <div style="margin-top:16px; display:flex; gap:8px; flex-wrap:wrap;">
            <button type="submit" class="btn btn-primary">✓ Submit</button>
            <button type="submit" name="action" value="none" class="btn btn-secondary">None relevant</button>
            <button type="submit" name="action" value="all" class="btn btn-secondary">All relevant</button>
            <button type="submit" name="action" value="skip" class="btn btn-secondary">Skip</button>
        </div>
    </form>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(render_page().encode())
    
    def do_POST(self):
        global CURRENT_IDX
        
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()
        params = parse_qs(body)
        
        action = params.get("action", ["submit"])[0]
        turn_idx = int(params.get("turn_idx", [CURRENT_IDX])[0])
        
        if action == "skip":
            CURRENT_IDX += 1
        elif action == "none":
            history_start = max(0, turn_idx - 15)
            LABELS.append({
                "current_turn": turn_idx,
                "current_text": TURNS[turn_idx]["content"][:500],
                "current_role": TURNS[turn_idx]["role"],
                "history_start": history_start,
                "history_end": turn_idx,
                "relevant_turns": [],
                "total_history": turn_idx - history_start,
                "num_relevant": 0,
            })
            save_labels()
            CURRENT_IDX += 1
        elif action == "all":
            history_start = max(0, turn_idx - 15)
            LABELS.append({
                "current_turn": turn_idx,
                "current_text": TURNS[turn_idx]["content"][:500],
                "current_role": TURNS[turn_idx]["role"],
                "history_start": history_start,
                "history_end": turn_idx,
                "relevant_turns": list(range(history_start, turn_idx)),
                "total_history": turn_idx - history_start,
                "num_relevant": turn_idx - history_start,
            })
            save_labels()
            CURRENT_IDX += 1
        else:
            # Regular submit with checkboxes
            relevant = [int(x) for x in params.get("relevant", [])]
            history_start = max(0, turn_idx - 15)
            LABELS.append({
                "current_turn": turn_idx,
                "current_text": TURNS[turn_idx]["content"][:500],
                "current_role": TURNS[turn_idx]["role"],
                "history_start": history_start,
                "history_end": turn_idx,
                "relevant_turns": relevant,
                "total_history": turn_idx - history_start,
                "num_relevant": len(relevant),
            })
            save_labels()
            CURRENT_IDX += 1
        
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress request logging


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m cortex_net.label_web <conversation.json> [--output labels.jsonl] [--port 8501]")
        sys.exit(1)
    
    CONV_PATH = sys.argv[1]
    OUTPUT_PATH = "relevance_labels.jsonl"
    port = 8501
    
    if "--output" in sys.argv:
        OUTPUT_PATH = sys.argv[sys.argv.index("--output") + 1]
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    
    load_state()
    
    print(f"🕸️ Labeling tool running at http://localhost:{port}")
    print(f"   Conversation: {CONV_PATH} ({len(TURNS)} turns)")
    print(f"   Output: {OUTPUT_PATH} ({len(LABELS)} existing labels)")
    print(f"   Starting at turn {CURRENT_IDX}")
    
    server = HTTPServer(("0.0.0.0", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n✅ {len(LABELS)} labels saved to {OUTPUT_PATH}")
