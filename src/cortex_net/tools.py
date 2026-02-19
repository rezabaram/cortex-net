"""Tool system for cortex-net agents.

Provides a registry of tools that agents can invoke via LLM function calling.
Tools are simple: name, description, parameters, and an execute function.

Built-in tools: file_read, file_write, file_list, shell_exec, web_search.
Custom tools can be registered by the agent or config.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class ToolParameter:
    """A single parameter for a tool."""

    name: str
    type: str  # string, integer, boolean, array
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass
class Tool:
    """A tool that an agent can invoke."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    execute: Callable[..., str] = lambda **kwargs: ""
    dangerous: bool = False  # requires confirmation in interactive mode

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def list_names(self) -> list[str]:
        return list(self.tools.keys())

    def to_openai_tools(self) -> list[dict]:
        """Get all tools in OpenAI function calling format."""
        return [t.to_openai_schema() for t in self.tools.values()]

    def execute(self, name: str, arguments: dict) -> str:
        """Execute a tool by name with arguments."""
        tool = self.tools.get(name)
        if not tool:
            return f"Error: Unknown tool '{name}'"
        try:
            return tool.execute(**arguments)
        except Exception as e:
            return f"Error executing {name}: {e}"


# --- Built-in tools ---

def _file_read(path: str, offset: int = 0, limit: int = 200) -> str:
    """Read a file's contents."""
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: {path} not found"
    if not p.is_file():
        return f"Error: {path} is not a file"
    try:
        lines = p.read_text().splitlines()
        selected = lines[offset:offset + limit]
        total = len(lines)
        result = "\n".join(selected)
        if offset + limit < total:
            result += f"\n\n... ({total - offset - limit} more lines. Use offset={offset + limit} to continue)"
        return result
    except Exception as e:
        return f"Error reading {path}: {e}"


def _file_write(path: str, content: str) -> str:
    """Write content to a file."""
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


def _file_list(path: str = ".", pattern: str = "*", recursive: bool = False) -> str:
    """List files in a directory."""
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: {path} not found"
    try:
        if recursive:
            files = sorted(p.rglob(pattern))
        else:
            files = sorted(p.glob(pattern))
        # Filter out hidden and common noise
        files = [f for f in files if not any(
            part.startswith('.') for part in f.relative_to(p).parts
        ) and '__pycache__' not in str(f)]
        result = "\n".join(str(f.relative_to(p)) for f in files[:100])
        if len(files) > 100:
            result += f"\n... ({len(files) - 100} more files)"
        return result or "(empty directory)"
    except Exception as e:
        return f"Error listing {path}: {e}"


def _shell_exec(command: str, workdir: str = ".", timeout: int = 30) -> str:
    """Execute a shell command."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n(stderr): {result.stderr}"
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def _file_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in a file."""
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: {path} not found"
    try:
        content = p.read_text()
        if old_text not in content:
            return f"Error: exact text not found in {path}"
        count = content.count(old_text)
        if count > 1:
            return f"Error: text appears {count} times in {path}. Be more specific."
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content)
        return f"Edited {path} (replaced {len(old_text)} chars with {len(new_text)} chars)"
    except Exception as e:
        return f"Error editing {path}: {e}"


def create_default_tools(workdir: str = ".") -> ToolRegistry:
    """Create a registry with all built-in tools."""
    registry = ToolRegistry()

    registry.register(Tool(
        name="file_read",
        description="Read a file's contents. Use offset/limit for large files.",
        parameters=[
            ToolParameter("path", "string", "Path to the file"),
            ToolParameter("offset", "integer", "Line to start from (0-indexed)", required=False),
            ToolParameter("limit", "integer", "Max lines to read (default 200)", required=False),
        ],
        execute=_file_read,
    ))

    registry.register(Tool(
        name="file_write",
        description="Write content to a file. Creates parent directories if needed.",
        parameters=[
            ToolParameter("path", "string", "Path to write to"),
            ToolParameter("content", "string", "Content to write"),
        ],
        execute=_file_write,
        dangerous=True,
    ))

    registry.register(Tool(
        name="file_edit",
        description="Replace exact text in a file. The old_text must match exactly.",
        parameters=[
            ToolParameter("path", "string", "Path to the file"),
            ToolParameter("old_text", "string", "Exact text to find"),
            ToolParameter("new_text", "string", "Text to replace with"),
        ],
        execute=_file_edit,
        dangerous=True,
    ))

    registry.register(Tool(
        name="file_list",
        description="List files in a directory.",
        parameters=[
            ToolParameter("path", "string", "Directory path", required=False),
            ToolParameter("pattern", "string", "Glob pattern (default '*')", required=False),
            ToolParameter("recursive", "boolean", "Search recursively", required=False),
        ],
        execute=_file_list,
    ))

    registry.register(Tool(
        name="shell",
        description="Execute a shell command. Use for running tests, git operations, etc.",
        parameters=[
            ToolParameter("command", "string", "Shell command to execute"),
            ToolParameter("workdir", "string", f"Working directory (default: {workdir})", required=False),
            ToolParameter("timeout", "integer", "Timeout in seconds (default 30)", required=False),
        ],
        execute=lambda command, workdir=workdir, timeout=30: _shell_exec(command, workdir, timeout),
        dangerous=True,
    ))

    return registry
