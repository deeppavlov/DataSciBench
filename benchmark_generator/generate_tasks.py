import argparse
import logging
from pathlib import Path

from .config import get_settings
from .llm import call_llm
from .models import TaskList

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_system_prompt() -> str:
    return (PROMPTS_DIR / "generate_tasks.md").read_text(encoding="utf-8")


def generate_tasks(codebase_path: Path, count: int, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    codebase_text = codebase_path.read_text(encoding="utf-8")
    system_prompt = load_system_prompt()
    user_prompt = (
        f"Generate exactly {count} Data Science benchmark tasks "
        f"based on the following codebase description:\n\n"
        f"---\n{codebase_text}\n---"
    )

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

    parser = argparse.ArgumentParser(description="Generate benchmark tasks from a codebase description")
    parser.add_argument("--codebase", type=Path, required=True, help="Path to codebase description text file")
    parser.add_argument("--count", type=int, default=5, help="Number of tasks to generate")
    parser.add_argument("--output_dir", type=Path, default=get_settings().output_dir, help="Output directory")
    args = parser.parse_args()

    dirs = generate_tasks(args.codebase, args.count, args.output_dir)
    print(f"Generated {len(dirs)} tasks in {args.output_dir}")
    for d in dirs:
        print(f"  {d.name}/")


if __name__ == "__main__":
    main()
