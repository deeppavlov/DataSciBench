import logging
from pathlib import Path

from ..core.llm import call_llm
from ..tools.solve_task import _run_script

logger = logging.getLogger(__name__)


def interactive_healthcheck(output_dir: Path, mcp_env: dict[str, str]) -> None:
    mcp_tools_path = output_dir / "_mcp_tools.py"
    api_doc_path = output_dir / "api_doc.md"

    if not mcp_tools_path.exists() or not api_doc_path.exists():
        print("Невозможно выполнить healthcheck: отсутствуют файлы _mcp_tools.py или api_doc.md")
        return

    api_doc = api_doc_path.read_text(encoding="utf-8")

    prompt = f"""
You are an expert diagnostic tool.
The following MCP tools are available:

{api_doc}

Your task is to write a single Python script that performs exactly ONE safe, read-only function call per available server to verify that the underlying systems are alive and responding.\x20
For example, you could use `list_allowed_directories()` for the filesystem and `list_databases()` for ClickHouse.

REQUIREMENTS:
1. Print "OK: <server_name>" if the call succeeds.
2. If a call fails, catch the exception and print "ERROR: <server_name> - <error details>".
3. The script must be self-contained.
4. CRITICAL: All tools are imported automatically into the global namespace. Note that these tools are ASYNCHRONOUS (`async def`). You MUST call them using `await` inside an `asyncio.run(main())` block. Call them directly (e.g. `await list_databases()`), DO NOT use server name prefixes (e.g. NOT `clickhouse.list_databases()`).
5. Return ONLY the raw python code. DO NOT include markdown formatting or explanations.
"""

    print("Генерация диагностического скрипта через LLM...")
    try:
        script_code = call_llm(system=prompt, user="Write the diagnostic script.").strip()
    except Exception as e:
        print(f"Ошибка при обращении к LLM: {e}")
        return

    script_code = script_code.removeprefix("```python")
    script_code = script_code.removeprefix("```")
    script_code = script_code.removesuffix("```")

    script_code = script_code.strip()

    script_path = output_dir / "mcp_healthcheck.py"
    script_path.write_text(script_code, encoding="utf-8")

    print("\n--- Запуск диагностического скрипта ---")
    try:
        output = _run_script(
            script_path, cwd=output_dir, code_mode=True, mcp_tools_path=mcp_tools_path, extra_env=mcp_env
        )
        print(output)
    except Exception as e:
        print(f"Ошибка при выполнении скрипта: {e}")
    print("---------------------------------------\n")
