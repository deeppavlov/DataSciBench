import asyncio
import subprocess
import sys
from dataclasses import dataclass

from metagpt.config2 import Config
from metagpt.logs import logger
from metagpt.provider.openai_api import OpenAILLM
from metagpt.utils.common import CodeParser

SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Write a complete, self-contained Python script that solves the given task.\n\n"
    "Rules:\n"
    "- The script must be runnable with `python solution.py`\n"
    "- Input data files are in the parent directory (`../`). Read them using `../filename`.\n"
    "- Save all output files to the current directory (`./`)\n"
    "- Include all necessary imports\n"
    "- Do not use interactive features (input(), plt.show(), etc.)\n"
    "- Use `matplotlib.use('Agg')` if you use matplotlib\n"
)

MAX_RETRIES = 5
CODE_TIMEOUT = 300


@dataclass
class SimpleRunResult:
    code: str
    output: str
    error_count: int


def _run_code(code: str, cwd: str) -> tuple[str, bool]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=CODE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: Script timed out after {CODE_TIMEOUT} seconds", False

    output = result.stdout
    if result.stderr:
        output += "\n--- stderr ---\n" + result.stderr
    success = result.returncode == 0
    if not success:
        output += f"\n[EXIT_CODE: {result.returncode}]"
    return output, success


async def run_task(requirement: str, config: Config) -> SimpleRunResult:
    llm = OpenAILLM(config.llm)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": requirement},
    ]

    logger.info("Generating initial solution...")
    response = await llm.acompletion_text(messages, stream=True)
    code = CodeParser.parse_code(block=None, text=response)

    messages.append({"role": "assistant", "content": response})

    import os
    cwd = os.getcwd()

    error_count = 0
    output = ""

    for attempt in range(MAX_RETRIES):
        logger.info(f"Running code (attempt {attempt + 1}/{MAX_RETRIES})...")
        output, success = _run_code(code, cwd)

        if success:
            logger.info("Code executed successfully.")
            break

        error_count += 1
        logger.warning(f"Attempt {attempt + 1} failed:\n{output}")

        if attempt + 1 >= MAX_RETRIES:
            logger.error("All retry attempts exhausted.")
            break

        fix_msg = (
            f"The script failed with the following output:\n\n"
            f"```\n{output}\n```\n\n"
            f"Fix the script. Return the complete corrected Python code."
        )
        messages.append({"role": "user", "content": fix_msg})

        logger.info("Requesting fix from LLM...")
        response = await llm.acompletion_text(messages, stream=True)
        code = CodeParser.parse_code(block=None, text=response)
        messages.append({"role": "assistant", "content": response})

    return SimpleRunResult(code=code, output=output, error_count=error_count)
