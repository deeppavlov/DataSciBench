import argparse
import logging
from pathlib import Path

from .config import get_settings
from .llm import call_llm
from .models import TaskList

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_system_prompt(code_mode: bool = False) -> str:
    name = "generate_tasks_code_mode.md" if code_mode else "generate_tasks.md"
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def generate_tasks(
    codebase_path: Path,
    count: int,
    output_dir: Path,
    topic_path: Path | None = None,
    code_mode: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    codebase_text = codebase_path.read_text(encoding="utf-8")
    system_prompt = load_system_prompt(code_mode)

    topic_text = ""
    if topic_path and topic_path.exists():
        topic_text = topic_path.read_text(encoding="utf-8")

    parts = [f"Generate exactly {count} benchmark tasks."]
    if topic_text:
        parts.append(f"Topic:\n---\n{topic_text}\n---")
    codebase_label = "Available API" if code_mode else "Codebase description"
    parts.append(f"{codebase_label}:\n---\n{codebase_text}\n---")

    user_prompt = "\n\n".join(parts)

    task_list: TaskList = call_llm(system_prompt, user_prompt, response_model=TaskList)

    existing = sorted(output_dir.glob("task_*"))
    next_num = 1
    if existing:
        last = existing[-1].name
        try:
            next_num = int(last.split("_")[1]) + 1
        except (IndexError, ValueError):
            pass

    created_dirs = []
    for i, task in enumerate(task_list.tasks):
        task_dir = output_dir / f"task_{next_num + i:03d}"
        task_dir.mkdir(parents=True, exist_ok=True)

        prompt_file = task_dir / "prompt.md"
        prompt_file.write_text(task.prompt, encoding="utf-8")

        if task.needs_input_data and task.input_data_code:
            input_data_file = task_dir / "input_data.py"
            input_data_file.write_text(task.input_data_code, encoding="utf-8")

        created_dirs.append(task_dir)
        logger.info("Created task: %s", task_dir.name)

    return created_dirs


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Generate benchmark tasks")
    parser.add_argument("--codebase", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output_dir", type=Path, default=settings.output_dir)
    parser.add_argument("--topic", type=Path, default=settings.topic_file)
    parser.add_argument("--code_mode", action="store_true")
    args = parser.parse_args()

    dirs = generate_tasks(args.codebase, args.count, args.output_dir, args.topic, args.code_mode)
    print(f"Generated {len(dirs)} tasks in {args.output_dir}")
    for d in dirs:
        print(f"  {d.name}/")


if __name__ == "__main__":
    main()
