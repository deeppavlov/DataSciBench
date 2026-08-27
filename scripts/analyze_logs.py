import json
import re
from collections import defaultdict
from typing import Any


def analyze_log(log_path):
    tasks: list[dict[str, Any]] = []
    current_task: dict[str, Any] | None = None

    # regex patterns
    cwd_pattern = re.compile(r"CWD: (.+)")
    attempt_pattern = re.compile(r"Running code \(attempt (\d+)/5\)")
    fail_pattern = re.compile(r"Attempt (\d+) failed:")
    exhausted_pattern = re.compile(r"All retry attempts exhausted")

    with open(log_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # New task start
        if "=== ENVIRONMENT DIAGNOSTICS ===" in line:
            if current_task:
                tasks.append(current_task)
            current_task = {
                "path": "unknown",
                "task_id": "unknown",
                "model": "unknown",
                "attempts": [],
                "final_status": "unknown",
            }
            # Look for CWD in next few lines
            for j in range(i + 1, min(i + 10, len(lines))):
                m = cwd_pattern.search(lines[j])
                if m:
                    full_path = m.group(1)
                    current_task["path"] = full_path
                    # Extract task_id and model from path
                    # Example: /.../data/human_8/gemma-3-27b-it_0
                    parts = full_path.split("/")
                    if len(parts) >= 2:
                        current_task["task_id"] = parts[-2]
                        current_task["model"] = parts[-1].rsplit("_", 1)[0]
                    break

        # New attempt
        m_att = attempt_pattern.search(line)
        if m_att and current_task:
            attempt_num = int(m_att.group(1))
            current_task["attempts"].append({"num": attempt_num, "status": "started", "error": None})

        # Attempt failure
        m_fail = fail_pattern.search(line)
        if m_fail and current_task:
            attempt_num = int(m_fail.group(1))
            # Find the attempt in current_task
            att_entry = next((a for a in current_task["attempts"] if a["num"] == attempt_num), None)
            if not att_entry:
                att_entry = {"num": attempt_num, "status": "failed", "error": ""}
                current_task["attempts"].append(att_entry)
            else:
                att_entry["status"] = "failed"
                att_entry["error"] = ""

            # Extract error trace until next log line (which starts with a date)
            j = i + 1
            error_lines = []
            while j < len(lines):
                if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", lines[j]):
                    break
                # Also stop if another task starts
                if "=== ENVIRONMENT DIAGNOSTICS ===" in lines[j]:
                    break
                error_lines.append(lines[j].strip())
                j += 1
            att_entry["error"] = "\n".join(error_lines).strip()

        # Exhausted
        if exhausted_pattern.search(line) and current_task:
            current_task["final_status"] = "failed"

        i += 1

    if current_task:
        tasks.append(current_task)

    return tasks


def summarize_errors(tasks):
    error_summary: defaultdict[str, int] = defaultdict(int)
    task_failures = []
    success_count = 0

    # Common exception patterns
    exception_re = re.compile(
        r"^([a-zA-Z0-9._]+Error|Exception|KeyError|IndexError|AttributeError|ImportError|ModuleNotFoundError|RuntimeError|ValueError|TypeError|NameError|SyntaxError|IndentationError): (.*)"
    )

    for task in tasks:
        if task.get("final_status") == "unknown":
            # If it's not marked failed and has at least one attempt, we assume it finished (or was interrupted)
            if any(a["status"] == "started" for a in task["attempts"]) and not any(
                a["status"] == "failed" for a in task["attempts"][-1:]
            ):
                success_count += 1
            elif not task["attempts"]:
                # Maybe it skipped/already existed
                pass
            # Check last attempt
            elif task["attempts"] and task["attempts"][-1]["status"] == "started":
                success_count += 1

        for att in task["attempts"]:
            if att["status"] == "failed" and att["error"]:
                err_text = att["error"]

                if "RateLimitError" in err_text:
                    error_summary["openai.RateLimitError"] += 1
                    continue

                found_exc = False
                lines = err_text.split("\n")
                for line in reversed(lines):
                    m = exception_re.match(line.strip())
                    if m:
                        exc_type = m.group(1)
                        error_summary[exc_type] += 1
                        found_exc = True
                        break

                if not found_exc:
                    for line in reversed(lines):
                        clean_line = line.strip()
                        if (
                            clean_line
                            and "[EXIT_CODE" not in clean_line
                            and "---" not in clean_line
                            and not any(x in clean_line for x in ["oneDNN", "TensorFlow", "binary is optimized"])
                        ):
                            error_summary[clean_line[:50]] += 1
                            found_exc = True
                            break

                if not found_exc:
                    error_summary["Unknown Python Error"] += 1

        if task.get("final_status") == "failed":
            task_failures.append(task)

    return error_summary, task_failures, success_count


if __name__ == "__main__":
    log_file = "output.log"
    results = analyze_log(log_file)
    summary, failures, successes = summarize_errors(results)

    print(f"Total tasks processed: {len(results)}")
    print(f"Tasks finished successfully (estimated): {successes}")
    print(f"Tasks failed (5 attempts exhausted): {len(failures)}")

    print("\nError Summary (by type, all attempts):")
    for err, count in sorted(summary.items(), key=lambda x: x[1], reverse=True):
        print(f"  {err:30}: {count}")

    print(f"\nDetails for {len(failures)} failed tasks:")
    for f in failures:
        last_error = "N/A"
        if f["attempts"]:
            for att in reversed(f["attempts"]):
                if att["status"] == "failed" and att["error"]:
                    err_lines = att["error"].split("\n")
                    for line in reversed(err_lines):
                        if any(x in line for x in ["Error:", "Exception:", "KeyError:"]):
                            last_error = line.strip()
                            break
                    if last_error == "N/A" and err_lines:
                        for line in reversed(err_lines):
                            if line.strip() and not any(
                                x in line for x in ["[EXIT_CODE", "---", "oneDNN", "TensorFlow"]
                            ):
                                last_error = line.strip()
                                break
                    break
        print(f"  Task: {f['task_id']:15} | Model: {f['model']:20} | Last Error: {last_error}")

    with open("analysis_data.json", "w") as out:
        json.dump(
            {
                "total": len(results),
                "successes": successes,
                "failures_count": len(failures),
                "summary": dict(summary),
                "failures": [
                    {"id": f["task_id"], "model": f["model"], "last_error": "..."}  # abbreviated
                    for f in failures
                ],
            },
            out,
            indent=2,
        )
