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
        [sys.executable, str(script_path.resolve())],
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
        "METRICS = []",
        "",
        "def metric(task_name, function, metric_name, ground_truth=None):",
        "    def decorator(func):",
        "        func.task_name = task_name",
        "        func.function = function",
        "        func.metric = metric_name",
        "        func.ground_truth = ground_truth",
        "        METRICS.append(func)",
        "        return func",
        "    return decorator",
        "",
    ]

    for m in metrics.metrics:
        gt = repr(m.ground_truth)
        decor = f"@metric(task_name={repr(m.task_name)}, function={repr(m.function)}, metric_name={repr(m.metric)}, ground_truth={gt})"
        parts.append(decor)
        code = m.code
        if r"\n" in code and "\n" not in code:
            code = code.replace(r"\n", "\n")
        parts.append(code)
        parts.append("")

    run_all_code = textwrap.dedent("""\
        def run_all():
            import os
            import traceback
            gt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt")
            passed = 0
            failed = 0
            for func in METRICS:
                name = func.metric
                gt_file = func.ground_truth
                gt_path = os.path.join(gt_dir, gt_file) if gt_file else None
                try:
                    result = func(gt_path)
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
            import sys
            ok = run_all()
            sys.exit(0 if ok else 1)
    """)
    parts.append(run_all_code)

    return "\n".join(parts)


def solve_single_task(task_dir: Path, codebase_text: str | None = None):
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

    system_prompt = _load_prompt("solve_task.md")
    
    user_prompt = prompt_text
    if codebase_text:
        user_prompt += (
            f"\n\nYou MUST use the following framework/codebase to solve this task:\n\n"
            f"---\n{codebase_text}\n---"
        )
        
    solve_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    solution, solve_messages = call_llm_multi(solve_messages, response_model=Solution)

    solution_path = task_dir / "solution.py"
    solution_path.write_text(solution.code, encoding="utf-8")
    logger.info("Saved solution.py")

    logger.info("Running solution.py...")
    exec_output = _run_script(solution_path, cwd=gt_dir)
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


def solve_tasks(output_dir: Path, codebase_text: str | None = None):
    task_dirs = sorted(d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("task_"))
    if not task_dirs:
        logger.warning("No task directories found in %s", output_dir)
        return

    for task_dir in task_dirs:
        if (task_dir / "solution.py").exists():
            logger.info("Skipping %s (already solved)", task_dir.name)
            continue
        solve_single_task(task_dir, codebase_text)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Solve benchmark tasks and generate gt + metrics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task_dir", type=Path, help="Path to a single task directory")
    group.add_argument("--output_dir", type=Path, help="Path to output directory (process all tasks)")
    parser.add_argument("--force", action="store_true", help="Re-solve already solved tasks")
    parser.add_argument("--codebase", type=Path, help="Path to codebase description to enforce usage of the framework")
    args = parser.parse_args()

    codebase_text = None
    if args.codebase and args.codebase.exists():
        codebase_text = args.codebase.read_text(encoding="utf-8")

    if args.task_dir:
        solve_single_task(args.task_dir, codebase_text)
    else:
        if args.force:
            task_dirs = sorted(d for d in args.output_dir.iterdir() if d.is_dir() and d.name.startswith("task_"))
            for td in task_dirs:
                solve_single_task(td, codebase_text)
        else:
            solve_tasks(args.output_dir, codebase_text)


if __name__ == "__main__":
    main()
