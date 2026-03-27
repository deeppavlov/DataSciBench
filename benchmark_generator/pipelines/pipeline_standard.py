import logging
from pathlib import Path

from ..core.config import get_settings
from ..tools.generate_tasks import generate_tasks
from ..tools.solve_task import solve_single_task, solve_tasks, _run_script
from ..tools.pack_task import pack_single_task, pack_all_tasks

logger = logging.getLogger(__name__)


def ask(prompt: str, default: str) -> str:
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    settings = get_settings()
    print("Интерактивный пайплайн Benchmark Generator (standard mode)")

    codebase = Path(ask("codebase", str(settings.codebase_file)))
    count = int(ask("count", "10"))
    output_dir = Path(ask("output_dir", str(settings.output_dir)))
    topic = Path(ask("topic", str(settings.topic_file)))

    print("Шаг 1: Создание промпта и input_data")
    dirs = generate_tasks(codebase, count, output_dir, topic_path=topic)
    print(f"Создано {len(dirs)} задач")
    for d in dirs:
        print(f"  {d.name}/")

    print("Шаг 1 завершен. Проверьте сгенерированные файлы.")

    work_mode = ask("1 - все задачи, 2 - конкретная", "2")

    if work_mode == "1":
        tasks = sorted(d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("task_"))
    else:
        task_name = ask("Имя задачи", "task_001")
        tasks = [output_dir / task_name]

    input("Нажмите Enter для запуска input_data (Шаг 2)...")
    print("Шаг 2: Запуск input_data")
    for task_dir in tasks:
        input_data = task_dir / "input_data.py"
        if input_data.exists():
            print(f"Запуск input_data.py в {task_dir}")
            _run_script(input_data, cwd=task_dir)
    print("Шаг 2 завершен.")

    codebase_text = codebase.read_text(encoding="utf-8") if codebase.exists() else None

    input("Нажмите Enter для генерации решений и метрик (Шаг 3)...")
    print("Шаг 3: solve_task")
    if work_mode == "1":
        solve_tasks(output_dir, codebase_text)
    else:
        solve_single_task(tasks[0], codebase_text)
    print("Шаг 3 завершен.")

    while True:
        print("-----------------------------------")
        print("1 - Повторная проверка (запустить solution.py и metrics.py)")
        print("2 - Перейти к упаковке")
        action = ask("Выберите", "2")

        if action == "1":
            for task_dir in tasks:
                solution = task_dir / "solution.py"
                metrics = task_dir / "metrics.py"
                if solution.exists() and metrics.exists():
                    gt_dir = task_dir / "gt"
                    gt_dir.mkdir(exist_ok=True)
                    print(f"Запуск solution.py в {task_dir}/gt...")
                    _run_script(solution, cwd=gt_dir)
                    print(f"Запуск metrics.py в {task_dir}/gt...")
                    out = _run_script(metrics, cwd=gt_dir)
                    print(out)
            print("Повторная проверка завершена.")
        else:
            break

    input("Нажмите Enter для упаковки (Шаг 4)...")
    print("Шаг 4: Упаковка")
    if work_mode == "1":
        prefix = ask("Prefix для задач", "gen")
        pack_all_tasks(output_dir, prefix, settings.benchmark_root)
    else:
        task_id = ask("task_id", "gen_001")
        pack_single_task(tasks[0], task_id, settings.benchmark_root)
    print("Шаг 4 завершен.")

    print("Пайплайн завершен!")


if __name__ == "__main__":
    main()
