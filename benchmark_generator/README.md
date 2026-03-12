# Automatic Benchmark Example Generation

A tool for the automatic creation of new DataSciBench tasks using an LLM.

## Requirements

```bash
pip install openai pydantic pydantic-settings pyyaml pytest
```

Set environment variable:
```bash
export OPENAI_API_KEY=sk-...
```

## Pipeline

### 1. Prompt Generation

```bash
python -m benchmark_generator.generate_tasks \
  --codebase ./context.txt \
  --count 5 \
  --output_dir ./benchmark_generator/output
```

Input: a text file describing the codebase. Output: directories `task_001/`, `task_002/`, ... containing `prompt.md` files (and `input_data.py` if the task requires input data).

After generation, you can:
- Read the prompts and remove unsuitable tasks
- Edit `prompt.md` in any editor

### 2. Task Solving

```bash
# All tasks in the directory:
python -m benchmark_generator.solve_task --output_dir ./benchmark_generator/output

# A single task:
python -m benchmark_generator.solve_task --task_dir ./benchmark_generator/output/task_001

# Recreate the solution (even if it already exists):
python -m benchmark_generator.solve_task --output_dir ./benchmark_generator/output --force
```

For each task, the following will be created:
- `solution.py` — solution code
- `gt/` — ground truth artifacts (tables, models, charts)
- `metrics.py` — metrics, can be read and run directly
- `chat_log.txt` — full dialogue with the model

### 3. Review

For each task:

1. Read `prompt.md` — is the task formulated properly?
2. Check `solution.py` — is the solution correct?
3. Verify artifacts in `gt/`
4. Read and run `metrics.py`:
   ```bash
   cd benchmark_generator/output/task_001
   python metrics.py
   ```
5. Edit and rerun if necessary

### 4. Packaging into the benchmark format

```bash
# All tasks:
python -m benchmark_generator.pack_task \
  --output_dir ./benchmark_generator/output \
  --prefix custom

# A single task:
python -m benchmark_generator.pack_task \
  --task_dir ./benchmark_generator/output/task_001 \
  --task_id custom_001
```

Result — tasks will appear in `data/` and `metric/` in the benchmark format.

## Settings

Via environment variables (prefix `BENCH_GEN_`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | API key (required) |
| `BENCH_GEN_LLM_MODEL` | `gpt-4o` | Model |
| `BENCH_GEN_LLM_API_BASE` | `https://api.openai.com/v1` | Base URL |
| `BENCH_GEN_CODE_TIMEOUT` | `300` | Code execution timeout (sec) |

## Tests

```bash
cd DataSciBench
pytest benchmark_generator/tests/ -v
```
