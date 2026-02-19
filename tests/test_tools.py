"""Tests for the tool system."""

import pytest
from pathlib import Path

from cortex_net.tools import (
    Tool, ToolParameter, ToolRegistry,
    create_default_tools,
    _file_read, _file_write, _file_edit, _file_list, _shell_exec,
)


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = Tool(name="test", description="A test tool")
        reg.register(tool)
        assert reg.get("test") is tool
        assert reg.get("nonexistent") is None

    def test_list_names(self):
        reg = create_default_tools()
        names = reg.list_names()
        assert "file_read" in names
        assert "file_write" in names
        assert "file_edit" in names
        assert "file_list" in names
        assert "shell" in names

    def test_openai_schema(self):
        reg = create_default_tools()
        schemas = reg.to_openai_tools()
        assert len(schemas) >= 5
        for s in schemas:
            assert s["type"] == "function"
            assert "name" in s["function"]
            assert "parameters" in s["function"]

    def test_execute(self):
        reg = ToolRegistry()
        reg.register(Tool(
            name="echo",
            description="Echo input",
            parameters=[ToolParameter("text", "string", "Text to echo")],
            execute=lambda text: f"Echo: {text}",
        ))
        result = reg.execute("echo", {"text": "hello"})
        assert result == "Echo: hello"

    def test_execute_unknown(self):
        reg = ToolRegistry()
        result = reg.execute("nonexistent", {})
        assert "Unknown tool" in result


class TestFileTools:
    def test_read(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        result = _file_read(str(f))
        assert "line1" in result
        assert "line2" in result

    def test_read_offset(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("\n".join(f"line{i}" for i in range(100)))
        result = _file_read(str(f), offset=50, limit=10)
        assert "line50" in result
        assert "line49" not in result

    def test_read_nonexistent(self):
        result = _file_read("/nonexistent/path")
        assert "Error" in result

    def test_write(self, tmp_path):
        f = tmp_path / "out.txt"
        result = _file_write(str(f), "hello world")
        assert "Written" in result
        assert f.read_text() == "hello world"

    def test_write_creates_dirs(self, tmp_path):
        f = tmp_path / "a" / "b" / "c.txt"
        _file_write(str(f), "deep")
        assert f.read_text() == "deep"

    def test_edit(self, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("Hello world, this is a test.")
        result = _file_edit(str(f), "world", "universe")
        assert "Edited" in result
        assert f.read_text() == "Hello universe, this is a test."

    def test_edit_not_found(self, tmp_path):
        f = tmp_path / "edit.txt"
        f.write_text("Hello world")
        result = _file_edit(str(f), "nonexistent text", "new")
        assert "Error" in result

    def test_list(self, tmp_path):
        (tmp_path / "a.py").touch()
        (tmp_path / "b.py").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.py").touch()
        result = _file_list(str(tmp_path))
        assert "a.py" in result
        assert "b.py" in result

    def test_list_recursive(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "deep.py").touch()
        result = _file_list(str(tmp_path), recursive=True)
        assert "deep.py" in result


class TestShellTool:
    def test_echo(self):
        result = _shell_exec("echo hello")
        assert "hello" in result

    def test_exit_code(self):
        result = _shell_exec("exit 1")
        assert "exit code: 1" in result

    def test_timeout(self):
        result = _shell_exec("sleep 10", timeout=1)
        assert "timed out" in result
