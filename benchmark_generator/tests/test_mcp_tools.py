import ast
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from benchmark_generator.core.mcp_tools import (
    _check_tool_name_collisions,
    _params_from_schema,
    generate_api_doc,
    generate_wrapper_module,
)


def _make_tool(name: str, description: str, input_schema: dict[str, Any]) -> Any:
    tool = MagicMock()
    tool.name = name
    tool.description = description
    tool.inputSchema = input_schema
    return tool


@pytest.fixture
def sample_tools() -> dict[str, list[Any]]:
    return {
        "test_server": [
            _make_tool(
                "query",
                "Execute a SQL query",
                {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string"},
                    },
                    "required": ["sql"],
                },
            ),
            _make_tool(
                "list_tables",
                "List all tables",
                {
                    "type": "object",
                    "properties": {},
                },
            ),
        ],
        "fs_server": [
            _make_tool(
                "read_file",
                "Read file content",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                },
            ),
            _make_tool(
                "write_file",
                "Write content to file",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            ),
        ],
    }


@pytest.fixture
def sample_mcp_config(tmp_path: Path) -> Path:
    config = {
        "mcpServers": {
            "test_server": {
                "command": "uvx",
                "args": ["test-mcp"],
            },
            "fs_server": {
                "command": "uvx",
                "args": ["fs-mcp", "--root", "/tmp"],
            },
        }
    }
    config_path = tmp_path / "mcp_servers.json"
    config_path.write_text(json.dumps(config))
    return config_path


class TestParamsFromSchema:
    def test_empty_schema(self) -> None:
        assert _params_from_schema({"type": "object", "properties": {}}) == []

    def test_required_params(self) -> None:
        params = _params_from_schema(
            {
                "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                "required": ["a"],
            }
        )
        assert len(params) == 2
        assert params[0] == ("a", "string", True)
        assert params[1] == ("b", "integer", False)


class TestGenerateApiDoc:
    def test_contains_all_tools(self, sample_tools: dict[str, list[Any]]) -> None:
        doc = generate_api_doc(sample_tools)
        assert "query" in doc
        assert "list_tables" in doc
        assert "read_file" in doc
        assert "write_file" in doc

    def test_contains_server_names(self, sample_tools: dict[str, list[Any]]) -> None:
        doc = generate_api_doc(sample_tools)
        assert "test_server" in doc
        assert "fs_server" in doc

    def test_contains_params(self, sample_tools: dict[str, list[Any]]) -> None:
        doc = generate_api_doc(sample_tools)
        assert "sql" in doc
        assert "path" in doc
        assert "content" in doc

    def test_contains_descriptions(self, sample_tools: dict[str, list[Any]]) -> None:
        doc = generate_api_doc(sample_tools)
        assert "Execute a SQL query" in doc
        assert "Read file content" in doc


class TestGenerateWrapperModule:
    def test_valid_python(self, sample_tools: dict[str, list[Any]], sample_mcp_config: Path) -> None:
        code = generate_wrapper_module(sample_tools, sample_mcp_config)
        ast.parse(code)

    def test_contains_functions(self, sample_tools: dict[str, list[Any]], sample_mcp_config: Path) -> None:
        code = generate_wrapper_module(sample_tools, sample_mcp_config)
        assert "def query(" in code
        assert "def list_tables(" in code
        assert "def read_file(" in code
        assert "def write_file(" in code

    def test_contains_server_params(self, sample_tools: dict[str, list[Any]], sample_mcp_config: Path) -> None:
        code = generate_wrapper_module(sample_tools, sample_mcp_config)
        assert "test-mcp" in code
        assert "fs-mcp" in code

    def test_functions_have_docstrings(self, sample_tools: dict[str, list[Any]], sample_mcp_config: Path) -> None:
        code = generate_wrapper_module(sample_tools, sample_mcp_config)
        assert "Execute a SQL query" in code
        assert "Read file content" in code

    def test_uses_persistent_loop(self, sample_tools: dict[str, list[Any]], sample_mcp_config: Path) -> None:
        code = generate_wrapper_module(sample_tools, sample_mcp_config)
        assert "atexit" in code
        assert "threading" in code
        assert "run_coroutine_threadsafe" in code

    def test_caches_sessions(self, sample_tools: dict[str, list[Any]], sample_mcp_config: Path) -> None:
        code = generate_wrapper_module(sample_tools, sample_mcp_config)
        assert "_sessions" in code
        assert "_get_session" in code


class TestToolNameCollisions:
    def test_no_collision_no_warning(
        self, sample_tools: dict[str, list[Any]], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            _check_tool_name_collisions(sample_tools)
        assert "collision" not in caplog.text.lower()

    def test_collision_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        tools = {
            "server_a": [_make_tool("query", "desc a", {"properties": {}})],
            "server_b": [_make_tool("query", "desc b", {"properties": {}})],
        }
        with caplog.at_level(logging.WARNING):
            _check_tool_name_collisions(tools)
        assert "collision" in caplog.text.lower()
        assert "query" in caplog.text
