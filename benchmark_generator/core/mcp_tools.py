import json
import logging
import textwrap
from pathlib import Path

from mcp.types import Tool

logger = logging.getLogger(__name__)


async def discover_tools(mcp_config_path: Path) -> dict[str, list[Tool]]:
    from pydantic_ai.mcp import load_mcp_servers

    servers = load_mcp_servers(str(mcp_config_path))
    result: dict[str, list[Tool]] = {}

    for server in servers:
        server_name = server.id or server.label
        async with server:
            tools = await server.list_tools()
            result[server_name] = tools
            logger.info("Server %s: %d tools", server_name, len(tools))
            for tool in tools:
                logger.info("  - %s: %s", tool.name, tool.description or "")

    return result


def _params_from_schema(schema: dict) -> list[tuple[str, str, bool]]:
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    params = []
    for name, info in props.items():
        typ = info.get("type", "Any")
        params.append((name, typ, name in required))
    return params


def generate_api_doc(tools: dict[str, list[Tool]]) -> str:
    lines = ["# Available API", ""]
    for server_name, server_tools in tools.items():
        lines.append(f"## {server_name}")
        lines.append("")
        for tool in server_tools:
            params = _params_from_schema(tool.inputSchema)
            sig_parts = []
            for pname, ptype, req in params:
                sig_parts.append(f"{pname}: {ptype}")
            sig = ", ".join(sig_parts)
            lines.append(f"### async {tool.name}({sig}) -> str")
            if tool.description:
                lines.append(tool.description)
            for pname, ptype, req in params:
                req_str = "required" if req else "optional"
                lines.append(f"- {pname} ({ptype}, {req_str})")
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def _check_tool_name_collisions(tools: dict[str, list[Tool]]):
    seen: dict[str, str] = {}
    for server_name, server_tools in tools.items():
        for tool in server_tools:
            if tool.name in seen:
                logger.warning(
                    "Tool name collision: '%s' exists in both '%s' and '%s'. "
                    "The second definition will overwrite the first in _mcp_tools.py.",
                    tool.name, seen[tool.name], server_name,
                )
            seen[tool.name] = server_name


def generate_wrapper_module(tools: dict[str, list[Tool]], mcp_config_path: Path) -> str:
    _check_tool_name_collisions(tools)

    config_data = json.loads(mcp_config_path.read_text(encoding="utf-8"))
    mcp_servers = config_data.get("mcpServers", {})

    header = textwrap.dedent("""\
        import asyncio
        import json
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

    """)

    servers_code = "_SERVERS = {\n"
    for name, cfg in mcp_servers.items():
        cmd = repr(cfg["command"])
        args = repr(cfg.get("args", []))
        env_part = ""
        if cfg.get("env"):
            env_part = f", env={repr(cfg['env'])}"
        servers_code += f"    {repr(name)}: StdioServerParameters(command={cmd}, args={args}{env_part}),\n"
    servers_code += "}\n\n"

    runtime = textwrap.dedent("""\
        import atexit
        import threading

        _loop = None
        _thread = None
        _sessions = {}
        _cleanup_tasks = []

        def _get_loop():
            global _loop, _thread
            if _loop is None:
                _loop = asyncio.new_event_loop()
                _thread = threading.Thread(target=_loop.run_forever, daemon=True)
                _thread.start()
                atexit.register(_shutdown)
            return _loop

        def _shutdown():
            global _loop, _thread
            if _loop is None:
                return
            for task in _cleanup_tasks:
                try:
                    asyncio.run_coroutine_threadsafe(task(), _loop).result(timeout=5)
                except Exception as e:
                    print(f"Warning: Exception during MCP cleanup: {e}")
            _loop.call_soon_threadsafe(_loop.stop)
            _thread.join(timeout=5)
            _loop = None
            _thread = None

        async def _get_session(server_name):
            if server_name not in _sessions:
                params = _SERVERS[server_name]
                ctx = stdio_client(params)
                read_stream, write_stream = await ctx.__aenter__()
                session = ClientSession(read_stream, write_stream)
                await session.__aenter__()
                await session.initialize()
                _sessions[server_name] = session

                async def _cleanup(s=session, c=ctx):
                    await s.__aexit__(None, None, None)
                    await c.__aexit__(None, None, None)
                _cleanup_tasks.append(_cleanup)
            return _sessions[server_name]

        async def _call(server_name, tool_name, args):
            session = await _get_session(server_name)
            args = {k: v for k, v in args.items() if v is not None}
            result = await session.call_tool(tool_name, args)
            text = "\\n".join(item.text if hasattr(item, "text") else str(item) for item in result.content)
            if result.isError:
                raise RuntimeError(f"Tool {tool_name} failed: {text}")
            return text

        async def _run_async(server_name, tool_name, args):
            loop = _get_loop()
            future = asyncio.run_coroutine_threadsafe(_call(server_name, tool_name, args), loop)
            return await asyncio.wrap_future(future)

    """)

    functions = []
    for server_name, server_tools in tools.items():
        for tool in server_tools:
            params = _params_from_schema(tool.inputSchema)
            sorted_params = sorted(params, key=lambda x: not x[2]) 
            sig_parts = []
            call_dict_parts = []
            for pname, ptype, _req in sorted_params:
                py_type = {"string": "str", "integer": "int", "number": "float", "boolean": "bool"}.get(ptype, "str")
                default = "" if _req else " = None"
                sig_parts.append(f"{pname}: {py_type}{default}")
                call_dict_parts.append(f"{repr(pname)}: {pname}")

            sig = ", ".join(sig_parts)
            call_dict = "{" + ", ".join(call_dict_parts) + "}"

            desc = tool.description or tool.name
            func_code = (
                f'async def {tool.name}({sig}) -> str:\n'
                f'    """{desc}"""\n'
                f'    return await _run_async({repr(server_name)}, {repr(tool.name)}, {call_dict})\n'
            )
            functions.append(func_code)

    return header + servers_code + runtime + "\n".join(functions) + "\n"
