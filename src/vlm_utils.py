import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE_DIR = os.path.join(ROOT_DIR, "evaluation_results", "vlm_bridge")
LOG_PATH = os.path.join(ROOT_DIR, "evaluation_results", "vlm_run_log.txt")
TIMEOUT_SECONDS = int(os.environ.get("VLM_BRIDGE_TIMEOUT", "300"))
JUDGE_MODEL = os.environ.get("VLM_JUDGE_MODEL", "sonnet")
JUDGE_EFFORT = os.environ.get("VLM_JUDGE_EFFORT", "high")

REWADRING_PROMPT = """Above are two figures, which are A and B. The first figure is the ground truth image and the second figure is the predicted image. The total score is 5. Please score B following the criteria below:
    - add 1 point for Data Representation Consistency: Ensure that the underlying data represented by the two charts is identical. This includes the values for all data points and the range of the data. Any variation in the dataset used would make the charts different.
    - add 1 point for Axis Labels and Scales: Verify that both charts have identical axis labels, units, and scales. Any difference in how the axes are labeled or scaled, such as using logarithmic vs. linear scales, can affect the interpretation of the data.
    - add 1 point for Graphical Elements: Check if the visual elements (such as lines, bars, markers, etc.) are represented the same way in both charts. Line thickness, marker styles, and colors should match across charts for them to be considered visually equal.
    - add 1 point for Legend and Annotations: Confirm that any legends, titles, or annotations (e.g., text labels, arrows, or highlights) are the same in both charts. These elements often provide crucial context for interpreting the chart.
    - add 1 point for Chart Dimensions and Layout: Ensure that the dimensions (height and width), aspect ratios, and layout of the charts are identical. Even if the content and representation are similar, a different aspect ratio or spacing between elements can change the chart's overall appearance and interpretation.

    Please write down the total score for B based on the criteria above, and provide a brief explanation of your reasoning. If you believe that the two figures are not identical, please explain the differences you observed.

    Your response should be in the following format:
    ### Explanation:
    ...
    ### Total Score:
    x/5
    """


def _digest(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _read_score(score_path: str) -> float | None:
    if not os.path.exists(score_path):
        return None
    try:
        with open(score_path) as f:
            return float(f.read().strip())
    except ValueError:
        return None


def _log(key: str, gt_abs: str, test_abs: str, score: float, source: str, detail: str | None = None) -> None:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{key}\tgt={gt_abs}\ttest={test_abs}\tscore={score}\tsource={source}\n")
        if detail:
            f.write(f"--- {source} ---\n{detail}\n")


def _parse_score(text: str | None) -> float | None:
    if not text:
        return None
    if "### Total Score:" in text:
        text = text.split("### Total Score:")[1]
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:/\s*5)?", text)
    if match is None:
        return None
    score = float(match.group(1))
    return score if 0 <= score <= 5 else None


def _find_cli() -> str | None:
    cli = os.environ.get("VLM_JUDGE_CLI")
    if cli:
        return cli
    cli = shutil.which("claude")
    if cli:
        return cli
    bundled = sorted(
        glob.glob(os.path.expanduser("~/.vscode*/extensions/anthropic.claude-code-*/resources/native-binary/claude"))
    )
    return bundled[-1] if bundled else None


def _bridge_vis_quality(gt_abs: str, test_abs: str, prompt: str, key: str) -> tuple[float, str]:
    cli = _find_cli()
    if cli is None:
        raise RuntimeError(
            "Судья по изображениям недоступен: не найден исполняемый файл claude. "
            "Укажите путь в переменной окружения VLM_JUDGE_CLI либо переключитесь на сетевого судью через VLM_JUDGE=api."
        )
    task = (
        f"Image A (ground truth): {gt_abs}\n"
        f"Image B (predicted): {test_abs}\n"
        "Read both image files with the Read tool, then do the task below.\n\n"
        f"{prompt}"
    )
    command = [cli, "-p", task, "--model", JUDGE_MODEL, "--effort", JUDGE_EFFORT, "--allowedTools", "Read"]
    result = subprocess.run(command, cwd=ROOT_DIR, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False)
    score = _parse_score(result.stdout)
    if score is None:
        raise RuntimeError(
            f"Судья по изображениям не вернул балл для заявки {key}: "
            f"код возврата {result.returncode}, вывод {result.stdout[-500:]!r}, ошибки {result.stderr[-500:]!r}"
        )
    return score, result.stdout


def _load_config() -> tuple[str, str]:
    try:
        from .vlm_config import API_KEY, BASE_URL
    except (ImportError, ValueError):
        from src.vlm_config import API_KEY, BASE_URL
    return API_KEY, BASE_URL


def _encode_image(image_path: str) -> str:
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _api_vis_quality(gt_abs: str, test_abs: str, prompt: str) -> tuple[float, str | None]:
    from openai import OpenAI

    api_key, base_url = _load_config()
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages: list[Any] = [
        {"role": "system", "content": "You are a helpful assistant that responds in Markdown."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image A:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_image(gt_abs)}"}},
                {"type": "text", "text": "Image B:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_image(test_abs)}"}},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    for _ in range(5):
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=1.0)
        content = response.choices[0].message.content
        score = _parse_score(content) if content and "### Total Score:" in content else None
        if score is not None:
            return score, content
    raise RuntimeError("Сетевой судья по изображениям не вернул строку '### Total Score:' за 5 попыток")


def vlm_vis_quality(ground_truth_path: str, test_path: str, prompt: str = REWADRING_PROMPT) -> float:
    gt_abs = os.path.abspath(ground_truth_path)
    test_abs = os.path.abspath(test_path)
    prompt_tag = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
    key = f"{_digest(gt_abs)}_{_digest(test_abs)}_{prompt_tag}"

    os.makedirs(BRIDGE_DIR, exist_ok=True)
    score_path = os.path.join(BRIDGE_DIR, f"{key}.score")

    cached = _read_score(score_path)
    if cached is not None:
        _log(key, gt_abs, test_abs, cached, "cache")
        return cached

    mode = os.environ.get("VLM_JUDGE", "bridge")
    if mode == "api":
        score, detail = _api_vis_quality(gt_abs, test_abs, prompt)
    else:
        req_path = os.path.join(BRIDGE_DIR, f"{key}.req.json")
        with open(req_path, "w") as f:
            json.dump(
                {
                    "key": key,
                    "ground_truth_image": gt_abs,
                    "predicted_image": test_abs,
                    "prompt": prompt,
                    "cwd": os.getcwd(),
                },
                f,
                indent=2,
            )
        score, detail = _bridge_vis_quality(gt_abs, test_abs, prompt, key)
        os.remove(req_path)

    with open(score_path, "w") as f:
        f.write(str(score))
    _log(key, gt_abs, test_abs, score, mode, detail)
    return score
