You are a benchmark evaluation expert. Given a solved Data Science task with its subtask decomposition and output files, generate metric functions.

## Context

Each metric is a Python function that checks if a specific output file is correct by comparing it against the ground truth version. The function:
- Takes one argument: `ground_truth` (path to the gt file)
- Reads the output file from the current directory
- Returns a boolean (True if correct, False otherwise)

## Metric types

Common metric patterns:
- **File existence**: `os.path.exists(filename)`
- **Shape comparison**: compare DataFrame shapes
- **Statistical comparison**: compare descriptive statistics
- **Value comparison**: compare specific values or ranges
- **Model accuracy**: load model, run predictions, check accuracy threshold
- **Completeness**: check that all expected columns/rows exist

## Output format

Return a JSON object with a list of metrics, each having:
- `task_name`: the subtask this metric evaluates (e.g. "Data cleaning and preprocessing")
- `function`: short description of what the metric checks
- `metric`: name of the metric
- `ground_truth`: filename of the ground truth file (from gt/ directory), or null if not needed
- `code`: the Python function as a string (must be named, take `ground_truth` as argument)
