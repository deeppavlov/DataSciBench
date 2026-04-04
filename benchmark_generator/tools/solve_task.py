import argparse
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from ..core.config import get_settings
from ..core.llm import call_llm_multi
from ..core.models import (
    InputDataCorrection,
    MetricsList,
    Solution,
    SolutionCorrection,
    SolutionMetricsCorrection,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _run_script(
    script_path: Path,
    cwd: Path,
    timeout: int | None = None,
    code_mode: bool = False,
    mcp_tools_path: Path | None = None,
    extra_env: dict | None = None,
) -> str:
    if timeout is None:
        timeout = get_settings().code_timeout

    env = None
    run_path = script_path.resolve()

    if code_mode and mcp_tools_path:
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        tools_dir = str(mcp_tools_path.parent.resolve())
        env["PYTHONPATH"] = tools_dir + ":" + env.get("PYTHONPATH", "")

        original_code = script_path.read_text(encoding="utf-8")
        wrapper_code = "from _mcp_tools import *\n" + original_code
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=str(cwd), delete=False)
        tmp.write(wrapper_code)
        tmp.flush()
        tmp.close()
        run_path = Path(tmp.name)
    elif extra_env:
        env = os.environ.copy()
        env.update(extra_env)

    try:
        result = subprocess.run(
            [sys.executable, str(run_path)],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Script %s timed out after %d seconds", script_path.name, timeout)
        return f"ERROR: Script timed out after {timeout} seconds"
    finally:
        if code_mode and mcp_tools_path and run_path != script_path.resolve():
            run_path.unlink(missing_ok=True)

    output = result.stdout
    if result.stderr:
        output += "\n--- stderr ---\n" + result.stderr
    if result.returncode != 0:
        logger.warning("Script %s exited with code %d", script_path.name, result.returncode)
    return output


def _append_chat_log(task_dir: Path, messages: list[dict]):
    log_path = task_dir / "chat_log.md"
    lines = []
    for msg in messages:
        role = msg["role"].upper()
        lines.append(f"=== {role} ===")
        lines.append(msg.get("content", ""))
        lines.append("")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _generate_metrics_py(metrics: MetricsList, prompt_text: str) -> str:
    parts = [
        '"""Auto-generated metrics. Run: python metrics.py"""',
        "",
        "import sys",
        "import os",
        'sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))',
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
            import sys
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


def run_and_fix_input_data(
    task_dir: Path,
    code_mode: bool = False,
    mcp_tools_path: Path | None = None,
    extra_env: dict | None = None,
) -> bool:
    input_data_path = task_dir / "input_data.py"
    prompt_path = task_dir / "prompt.md"

    if not input_data_path.exists():
        return True

    messages = [
        {
            "role": "system",
            "content": "You are an expert debugger. The script input_data.py failed to prepare the environment. Fix either the task prompt or input_data.py. If no changes are needed for a specific field, leave it null.",
        }
    ]

    for attempt in range(5):
        out = _run_script(
            input_data_path,
            cwd=task_dir,
            code_mode=code_mode,
            mcp_tools_path=mcp_tools_path,
            extra_env=extra_env,
        )

        if "exited with code" not in out and "Traceback" not in out and "ERROR:" not in out:
            return True

        prompt_text = prompt_path.read_text(encoding="utf-8")
        code_text = input_data_path.read_text(encoding="utf-8")

        user_msg = f"Task prompt:\n{prompt_text}\n\ninput_data.py:\n{code_text}\n\nExecution Output/Error:\n{out}\n\nProvide fixes."
        messages.append({"role": "user", "content": user_msg})

        correction, messages = call_llm_multi(messages, response_model=InputDataCorrection)
        _append_chat_log(task_dir, [{"role": "user", "content": user_msg}, {"role": "assistant", "content": correction.model_dump_json()}])

        if correction.updated_prompt:
            prompt_path.write_text(correction.updated_prompt, encoding="utf-8")
        if correction.updated_input_data_code:
            input_data_path.write_text(correction.updated_input_data_code, encoding="utf-8")

    return False


def solve_single_task(
    task_dir: Path,
    codebase_text: str | None = None,
    code_mode: bool = False,
    mcp_tools_path: Path | None = None,
    mcp_env: dict | None = None,
):
    prompt_path = task_dir / "prompt.md"
    if not prompt_path.exists():
        return

    logger.info("Solving task: %s", task_dir.name)
    
    gt_dir = task_dir / "gt"
    gt_dir.mkdir(exist_ok=True)
    solution_path = task_dir / "solution.py"

    prompt_text = prompt_path.read_text(encoding="utf-8")
    prompt_name = "solve_task_code_mode.md" if code_mode else "solve_task.md"
    system_prompt = _load_prompt(prompt_name)
    user_prompt = prompt_text

    if codebase_text:
        label = "Available API" if code_mode else "Framework/codebase"
        user_prompt += f"\n\nYou MUST use the following {label} to solve this task:\n\n---\n{codebase_text}\n---"

    solve_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    solution, solve_messages = call_llm_multi(solve_messages, response_model=Solution)
    _append_chat_log(task_dir, solve_messages)

    solution_path.write_text(solution.code, encoding="utf-8")

    exec_output = ""
    for attempt in range(5):
        exec_output = _run_script(
            solution_path, cwd=gt_dir, code_mode=code_mode, mcp_tools_path=mcp_tools_path, extra_env=mcp_env
        )
        if "exited with code" not in exec_output and "Traceback" not in exec_output and "ERROR:" not in exec_output:
            break

        user_msg = f"solution.py failed during execution:\n{exec_output}\nFix the code or subtasks. Return null for fields that do not require changes."
        solve_messages.append({"role": "user", "content": user_msg})
        correction, solve_messages = call_llm_multi(solve_messages, response_model=SolutionCorrection)
        _append_chat_log(task_dir, [{"role": "user", "content": user_msg}, {"role": "assistant", "content": correction.model_dump_json()}])

        if correction.updated_code:
            solution.code = correction.updated_code
            solution_path.write_text(solution.code, encoding="utf-8")
        if correction.updated_subtasks:
            solution.subtasks = correction.updated_subtasks

    generated_files = [f.name for f in gt_dir.iterdir() if f.is_file()]

    metric_messages = [
        {"role": "system", "content": _load_prompt("generate_metrics.md")},
        {
            "role": "user",
            "content": (
                f"Here is the task prompt:\n\n{prompt_text}\n\n"
                f"The solution has been executed. Output:\n\n```\n{exec_output}\n```\n\n"
                f"Generated files in gt/: {generated_files}\n\n"
                f"Subtask decomposition:\n"
                + "\n".join(f"- {st.name}: {st.description} -> {st.output_files}" for st in solution.subtasks)
                + "\n\nGenerate metrics for each subtask."
            ),
        },
    ]
    metrics, metric_messages = call_llm_multi(metric_messages, response_model=MetricsList)
    _append_chat_log(task_dir, metric_messages)

    metrics_path = task_dir / "metrics.py"
    metrics_path.write_text(_generate_metrics_py(metrics, prompt_text), encoding="utf-8")

    for attempt in range(5):
        metrics_output = _run_script(metrics_path, cwd=gt_dir, extra_env=mcp_env)
        
        if "FAIL:" not in metrics_output and "ERROR:" not in metrics_output and "exited with code" not in metrics_output and "Traceback" not in metrics_output:
            verify_log = task_dir / "verify_log.txt"
            verify_log.write_text(metrics_output, encoding="utf-8")
            break

        user_msg = f"metrics.py failed or found incorrect solution outputs:\n{metrics_output}\nFix the solution code and/or the metrics. Return null for fields that do not require changes."
        metric_messages.append({"role": "user", "content": user_msg})
        correction, metric_messages = call_llm_multi(metric_messages, response_model=SolutionMetricsCorrection)
        _append_chat_log(task_dir, [{"role": "user", "content": user_msg}, {"role": "assistant", "content": correction.model_dump_json()}])

        if correction.updated_solution_code:
            solution.code = correction.updated_solution_code
            solution_path.write_text(solution.code, encoding="utf-8")
            exec_output = _run_script(
                solution_path, cwd=gt_dir, code_mode=code_mode, mcp_tools_path=mcp_tools_path, extra_env=mcp_env
            )
            _append_chat_log(task_dir, [{"role": "system", "content": f"Re-ran updated solution. Output:\n{exec_output}"}])

        if correction.updated_metrics:
            metrics.metrics = correction.updated_metrics
            metrics_path.write_text(_generate_metrics_py(metrics, prompt_text), encoding="utf-8")


def solve_tasks(
    output_dir: Path,
    codebase_text: str | None = None,
    code_mode: bool = False,
    mcp_tools_path: Path | None = None,
    mcp_env: dict | None = None,
):
    task_dirs = sorted(d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("task_"))
    for task_dir in task_dirs:
        if (task_dir / "solution.py").exists():
            continue
        solve_single_task(task_dir, codebase_text, code_mode, mcp_tools_path, mcp_env)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task_dir", type=Path)
    group.add_argument("--output_dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--codebase", type=Path)
    parser.add_argument("--code_mode", action="store_true")
    parser.add_argument("--mcp_tools_path", type=Path)
    args = parser.parse_args()

    codebase_text = None
    if args.codebase and args.codebase.exists():
        codebase_text = args.codebase.read_text(encoding="utf-8")

    if args.task_dir:
        solve_single_task(args.task_dir, codebase_text, args.code_mode, args.mcp_tools_path)
    else:
        if args.force:
            task_dirs = sorted(d for d in args.output_dir.iterdir() if d.is_dir() and d.name.startswith("task_"))
            for td in task_dirs:
                solve_single_task(td, codebase_text, args.code_mode, args.mcp_tools_path)
        else:
            solve_tasks(args.output_dir, codebase_text, args.code_mode, args.mcp_tools_path)


if __name__ == "__main__":
    main()