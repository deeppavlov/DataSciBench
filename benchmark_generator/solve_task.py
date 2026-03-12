import argparse
import json
import logging
import subprocess
import sys
import textwrap
from pathlib import Path

from .config import get_settings
from .llm import call_llm, call_llm_multi
from .models import MetricsList, Solution

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _run_script(script_path: Path, cwd: Path, timeout: int | None = None) -> str:
    if timeout is None:
        timeout = get_settings().code_timeout
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    output = result.stdout
    if result.stderr:
        output += "\n--- stderr ---\n" + result.stderr
    if result.returncode != 0:
        logger.warning("Script %s exited with code %d", script_path.name, result.returncode)
    return output


def _save_chat_log(task_dir: Path, messages: list[dict]):
    log_path = task_dir / "chat_log.txt"
    lines = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        lines.append(f"=== {role} ===")
        lines.append(content)
        lines.append("")
    log_path.write_text("\n".join(lines), encoding="utf-8")


def _generate_metrics_py(metrics: MetricsList, prompt_text: str) -> str:
    parts = [
        '"""Auto-generated metrics. Run: python metrics.py"""',
        "",
        "import os",
        "import sys",
        "import traceback",
        "",
        "",
        "METRICS = [",
    ]

    for m in metrics.metrics:
        gt = repr(m.ground_truth)
        parts.append("    {")
        parts.append(f"        \"task_name\": {repr(m.task_name)},")
        parts.append(f"        \"function\": {repr(m.function)},")
        parts.append(f"        \"metric\": {repr(m.metric)},")
        parts.append(f"        \"ground_truth\": {gt},")
        parts.append(f"        \"code\": {repr(m.code)},")
        parts.append("    },")

    parts.append("]")
    parts.append("")
    parts.append("")

    run_all_code = textwrap.dedent("""\
        def run_all():
            gt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt")
            passed = 0
            failed = 0
            for entry in METRICS:
                name = entry["metric"]
                gt_file = entry.get("ground_truth")
                gt_path = os.path.join(gt_dir, gt_file) if gt_file else None
                code = entry["code"]
                try:
                    local_ns = {}
                    exec(code, {}, local_ns)
                    func = list(local_ns.values())[0]
                    result = func(gt_path) if gt_path else func()
                    if result:
                        print(f"  PASS: {name}")
                        passed += 1
                    else:
                        print(f"  FAIL: {name}")
                        failed += 1
                except Exception:
                    print(f"  ERROR: {name}")
                    traceback.print_exc()
                    failed += 1
            print(f"\\nResults: {passed} passed, {failed} failed")
            return failed == 0


        if __name__ == "__main__":
            ok = run_all()
            sys.exit(0 if ok else 1)
    """)
    parts.append(run_all_code)

    return "\n".join(parts)


def solve_single_task(task_dir: Path):
    prompt_path = task_dir / "prompt.md"
    if not prompt_path.exists():
        logger.warning("No prompt.md in %s, skipping", task_dir)
        return

    logger.info("Solving task: %s", task_dir.name)
    prompt_text = prompt_path.read_text(encoding="utf-8")

    gt_dir = task_dir / "gt"
    gt_dir.mkdir(exist_ok=True)

    input_data_path = task_dir / "input_data.py"
    if input_data_path.exists():
        logger.info("Running input_data.py...")
        output = _run_script(input_data_path, cwd=task_dir)
        logger.debug("input_data.py output: %s", output)

    solve_messages = [
        {"role": "system", "content": _load_prompt("solve_task.md")},
        {"role": "user", "content": prompt_text},
    ]
    solution, solve_messages = call_llm_multi(solve_messages, response_model=Solution)

    solution_path = task_dir / "solution.py"
    solution_path.write_text(solution.code, encoding="utf-8")
    logger.info("Saved solution.py")

    logger.info("Running solution.py...")
    exec_output = _run_script(solution_path, cwd=task_dir)
    logger.info("Solution output:\n%s", exec_output)

    generated_files = [f.name for f in gt_dir.iterdir() if f.is_file()]

    metric_messages = [
        {"role": "system", "content": _load_prompt("generate_metrics.md")},
        {
            "role": "user",
            "content": (
                f"Here is the task prompt:\n\n{prompt_text}\n\n"
                f"The solution has been executed. Output:\n\n"
                f"```\n{exec_output}\n```\n\n"
                f"Generated files in gt/: {generated_files}\n\n"
                f"Subtask decomposition:\n"
                + "\n".join(
                    f"- {st.name}: {st.description} -> {st.output_files}"
                    for st in solution.subtasks
                )
                + "\n\nGenerate metrics for each subtask."
            ),
        },
    ]
    metrics, metric_messages = call_llm_multi(metric_messages, response_model=MetricsList)

    metrics_py_content = _generate_metrics_py(metrics, prompt_text)
    metrics_path = task_dir / "metrics.py"
    metrics_path.write_text(metrics_py_content, encoding="utf-8")
    logger.info("Saved metrics.py")

    all_messages = solve_messages + [{"role": "system", "content": "--- METRICS GENERATION PHASE ---"}] + metric_messages
    _save_chat_log(task_dir, all_messages)
    logger.info("Saved chat_log.txt")

    logger.info("Running metrics.py to verify...")
    metrics_output = _run_script(metrics_path, cwd=task_dir)
    verify_log = task_dir / "verify_log.txt"
    verify_log.write_text(metrics_output, encoding="utf-8")
    logger.info("Verification:\n%s", metrics_output)


def solve_tasks(output_dir: Path):
    task_dirs = sorted(d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("task_"))
    if not task_dirs:
        logger.warning("No task directories found in %s", output_dir)
        return

    for task_dir in task_dirs:
        if (task_dir / "solution.py").exists():
            logger.info("Skipping %s (already solved)", task_dir.name)
            continue
        solve_single_task(task_dir)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Solve benchmark tasks and generate gt + metrics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task_dir", type=Path, help="Path to a single task directory")
    group.add_argument("--output_dir", type=Path, help="Path to output directory (process all tasks)")
    parser.add_argument("--force", action="store_true", help="Re-solve already solved tasks")
    args = parser.parse_args()

    if args.task_dir:
        solve_single_task(args.task_dir)
    else:
        if args.force:
            task_dirs = sorted(d for d in args.output_dir.iterdir() if d.is_dir() and d.name.startswith("task_"))
            for td in task_dirs:
                solve_single_task(td)
        else:
            solve_tasks(args.output_dir)


if __name__ == "__main__":
    main()
