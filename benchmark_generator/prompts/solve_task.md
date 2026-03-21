You are a Data Science expert. You will receive a task description (prompt) and must solve it completely.

## Requirements

1. Write a single Python script that solves the entire task
2. The script must be self-contained and runnable with `python solution.py`
3. Your script will be executed from within a `gt/` subdirectory, but the input data files are located in the parent directory (`../`). You must read any inputs using `../` (e.g., `pd.read_csv("../data.csv")`).
4. All output files must be saved in the current working directory (`./`), which is the `gt/` subdirectory.
5. Use standard libraries: numpy, pandas, scikit-learn, matplotlib, seaborn, tensorflow/pytorch as needed
6. The script should handle all steps described in the prompt

## Output format

Return a JSON object with:
- `code`: the complete Python script as a string
- `subtasks`: a list of subtasks, each with:
  - `name`: short name of the subtask (e.g. "Data cleaning and preprocessing")
  - `description`: what this subtask does
  - `output_files`: list of filenames this subtask produces

The subtask decomposition is critical — it will be used to generate metrics for each step.
