import asyncio
import json
import logging
import shutil
from pathlib import Path

from ..core.config import get_settings
from ..core.mcp_tools import discover_tools, generate_api_doc, generate_wrapper_module
from ..tools.generate_tasks import generate_tasks
from ..tools.pack_task import pack_all_tasks, pack_single_task
from ..tools.solve_task import _run_script, solve_single_task, solve_tasks

logger = logging.getLogger(__name__)


def ask(prompt: str, default: str) -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


async def run_pipeline():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    settings = get_settings()
    print("Интерактивный пайплайн Benchmark Generator (code mode)")

    mcp_config = Path(ask("mcp_servers.json", str(settings.mcp_config)))
    topic = Path(ask("topic", str(settings.topic_file)))
    count = int(ask("count", "10"))
    output_dir = Path(ask("output_dir", str(settings.output_dir)))

    print("\n=== Шаг 0: Подключение к MCP-серверам ===")
    tools = await discover_tools(mcp_config)

    total_tools = sum(len(t) for t in tools.values())
    print(f"Обнаружено {total_tools} инструментов из {len(tools)} серверов")

    api_doc = generate_api_doc(tools)
    api_doc_path = output_dir / "api_doc.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    api_doc_path.write_text(api_doc, encoding="utf-8")
    print(f"Сохранён api_doc.md:\n{api_doc}")

    wrapper_code = generate_wrapper_module(tools, mcp_config)
    mcp_tools_path = output_dir / "_mcp_tools.py"
    mcp_tools_path.write_text(wrapper_code, encoding="utf-8")
    print(f"Сохранён _mcp_tools.py ({len(wrapper_code)} bytes)")

    mcp_env = {}
    if mcp_config.exists():
        with open(mcp_config, encoding="utf-8") as f:
            data = json.load(f)
            for srv in data.get("mcpServers", {}).values():
                mcp_env.update(srv.get("env", {}))

    print("\n=== Шаг 1: Генерация задач ===")
    dirs = generate_tasks(api_doc_path, count, output_dir, topic_path=topic, code_mode=True, env_hints=mcp_env)
    print(f"Создано {len(dirs)} задач")
    for d in dirs:
        print(f"  {d.name}/")

    input("\nПроверьте задачи. Нажмите Enter для продолжения...")

    work_mode = ask("1 - все задачи, 2 - конкретная", "2")

    if work_mode == "1":
        tasks = sorted(d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("task_"))
    else:
        task_name = ask("Имя задачи", "task_001")
        tasks = [output_dir / task_name]

    print("\n=== Шаг 2: Запуск input_data ===")
    for task_dir in tasks:
        input_data = task_dir / "input_data.py"
        if input_data.exists():
            print(f"Запуск input_data.py в {task_dir}")
            out = _run_script(input_data, cwd=task_dir, code_mode=True, mcp_tools_path=mcp_tools_path, extra_env=mcp_env)
            if "exited with code" in out or "ERROR" in out:
                print(out)
    print("Шаг 2 завершен.")

    api_doc_text = api_doc_path.read_text(encoding="utf-8")

    input("\nНажмите Enter для генерации решений (Шаг 3)...")
    print("\n=== Шаг 3: Решение задач ===")
    if work_mode == "1":
        solve_tasks(output_dir, api_doc_text, code_mode=True, mcp_tools_path=mcp_tools_path, mcp_env=mcp_env)
    else:
        solve_single_task(tasks[0], api_doc_text, code_mode=True, mcp_tools_path=mcp_tools_path, mcp_env=mcp_env)
    print("Шаг 3 завершен.")

    while True:
        print("-----------------------------------")
        print("1 - Повторная проверка")
        print("2 - Перейти к упаковке")
        action = ask("Выберите", "2")

        if action == "1":
            for task_dir in tasks:
                input_data = task_dir / "input_data.py"
                if input_data.exists():
                    print(f"Сброс среды: запуск input_data.py в {task_dir}...")
                    _run_script(input_data, cwd=task_dir, code_mode=True, mcp_tools_path=mcp_tools_path, extra_env=mcp_env)
                solution = task_dir / "solution.py"
                metrics = task_dir / "metrics.py"
                if solution.exists() and metrics.exists():
                    gt_dir = task_dir / "gt"
                    gt_dir.mkdir(exist_ok=True)
                    print(f"Запуск solution.py в {task_dir}/gt...")
                    _run_script(solution, cwd=gt_dir, code_mode=True, mcp_tools_path=mcp_tools_path, extra_env=mcp_env)
                    print(f"Запуск metrics.py в {task_dir}/gt...")
                    out = _run_script(metrics, cwd=gt_dir, extra_env=mcp_env)
                    print(out)
            print("Повторная проверка завершена.")
        else:
            break

    input("\nНажмите Enter для упаковки (Шаг 4)...")
    print("\n=== Шаг 4: Упаковка ===")
    if work_mode == "1":
        prefix = ask("Prefix для задач", "code_gen")
        pack_all_tasks(output_dir, prefix, settings.benchmark_root)
    else:
        task_id = ask("task_id", "code_gen_001")
        pack_single_task(tasks[0], task_id, settings.benchmark_root)
    print("Шаг 4 завершен.")

    dest = settings.benchmark_root / "code_mode" / "_mcp_tools.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mcp_tools_path, dest)
    print(f"Скопирован _mcp_tools.py -> {dest}")

    print("Пайплайн завершен!")


def main():
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
