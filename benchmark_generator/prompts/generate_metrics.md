You are a benchmark evaluation expert. Given a solved Data Science task with its subtask decomposition and output files, generate metric functions.

## Context

Each metric is a Python function that checks if a specific output file is correct by comparing it against the ground truth version. The function:
- Takes one argument: `ground_truth` (path to the gt file)
- Reads the output file from the current directory
- Returns a boolean (True if correct, False otherwise)
- **CRITICAL**: All necessary imports (e.g., `import os`, `import pandas as pd`) MUST be placed **inside** the function body. Do not put imports at the top of the code snippet.

## Metric types

Common metric patterns:
- **File existence**: `os.path.exists(filename)`
- **Shape comparison**: compare DataFrame shapes
- **Statistical comparison**: compare descriptive statistics
- **Value comparison**: compare specific values or ranges
- **Model accuracy**: load model, run predictions, check accuracy threshold
- **Completeness**: check that all expected columns/rows exist
- **VLM-as-a-judge (Visualization Quality)**: for generated plots (e.g., PNG). Use `from src.vlm_utils import vlm_vis_quality` and return `float(vlm_vis_quality(ground_truth, "output.png"))`.
- **LLM-as-a-judge (Text Quality)**: for open-ended text answers, analysis, or reports (e.g., MD, TXT). Use `from src.llm_utils import llm_text_quality` and return `float(llm_text_quality(ground_truth, "output.txt"))`.

## Output format

Return a JSON object with a list of metrics, each having:
- `task_name`: the subtask this metric evaluates (e.g. "Data cleaning and preprocessing")
- `function`: short description of what the metric checks
- `metric`: name of the metric
- `ground_truth`: filename of the ground truth file (from gt/ directory), or null if not needed
- `code`: the Python function as a string (must be named, take `ground_truth` as argument)
