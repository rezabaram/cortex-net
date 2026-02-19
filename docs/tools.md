# Tool System

cortex-net agents can invoke tools via LLM function calling. The tool system is provider-agnostic — any LLM that supports OpenAI-compatible function calling can use it.

## How It Works

```
User message
    → cortex-net assembles context (memories, strategy, confidence)
    → LLM receives context + tool definitions
    → LLM decides to call a tool (or respond directly)
    → Tool executes, result sent back to LLM
    → LLM calls more tools or gives final response
    → (up to max_tool_rounds iterations)
```

The LLM drives the tool loop. cortex-net provides the context; the LLM decides what to do with it.

## Built-in Tools

### file_read
Read a file's contents with optional offset/limit for large files.
```
file_read(path, offset=0, limit=200)
```

### file_write
Write content to a file. Creates parent directories if needed.
```
file_write(path, content)
```

### file_edit
Replace exact text in a file. The old text must match exactly (like a surgical edit).
```
file_edit(path, old_text, new_text)
```

### file_list
List files in a directory with optional glob pattern.
```
file_list(path=".", pattern="*", recursive=False)
```

### shell
Execute a shell command with timeout.
```
shell(command, workdir=".", timeout=30)
```

## Enabling Tools

In `AgentConfig`:
```python
config = AgentConfig(
    tools_enabled=True,
    tools_workdir="/path/to/project",
    max_tool_rounds=10,
    ...
)
```

In agent `config.json`:
```json
{
  "tools_enabled": true,
  "tools_workdir": "/home/user/project"
}
```

## Custom Tools

Register custom tools via the `ToolRegistry`:

```python
from cortex_net.tools import Tool, ToolParameter, ToolRegistry

registry = ToolRegistry()

registry.register(Tool(
    name="search_code",
    description="Search for a pattern across all source files",
    parameters=[
        ToolParameter("pattern", "string", "Regex pattern to search for"),
        ToolParameter("path", "string", "Directory to search", required=False),
    ],
    execute=lambda pattern, path=".": _my_search_fn(pattern, path),
))

# Assign to agent
agent.tool_registry = registry
```

### Tool Schema

Each tool has:
- **name** — unique identifier
- **description** — what the tool does (shown to the LLM)
- **parameters** — typed inputs with descriptions
- **execute** — Python callable that returns a string result
- **dangerous** — flag for tools that modify state (for future confirmation UIs)

Tools are automatically converted to OpenAI function calling format:
```python
registry.to_openai_tools()
# → [{"type": "function", "function": {"name": "file_read", ...}}, ...]
```

## Tool Loop

When tools are enabled, `chat()` runs an iterative loop:

```
1. Send messages + tool definitions to LLM
2. If LLM returns tool_calls:
   a. Execute each tool call
   b. Append results as tool messages
   c. Send everything back to LLM
   d. Repeat from step 2
3. If LLM returns a text response (no tool_calls):
   → Return as the final answer
4. If max_tool_rounds reached:
   → Return whatever the LLM last said
```

Tool output is capped at 4KB per call to stay within context limits.

## Safety

- **Timeout:** Shell commands have a default 30-second timeout
- **Max rounds:** Tool loop limited to `max_tool_rounds` (default 10) to prevent runaway loops
- **Output cap:** Tool results truncated to 4KB
- **Dangerous flag:** Tools that modify state are marked `dangerous=True` (for future confirmation flows)
- **Working directory:** Tools are scoped to `tools_workdir` by default

## Example: Atlas as a Developer Agent

Atlas is configured with tools enabled and working directory set to the cortex-net repo:

```json
{
  "name": "Atlas",
  "tools_enabled": true,
  "tools_workdir": "/home/reza/workstation/cortex-net",
  "system_prompt": "You are Atlas, a developer agent for cortex-net..."
}
```

Atlas can:
- Read source code: `file_read("src/cortex_net/memory_gate.py")`
- Run tests: `shell("python -m pytest tests/ -q")`
- Make edits: `file_edit("src/cortex_net/agent.py", "old code", "new code")`
- Explore the project: `file_list("src/cortex_net", recursive=True)`
- Commit changes: `shell("git add -A && git commit -m 'fix: ...' && git push")`
