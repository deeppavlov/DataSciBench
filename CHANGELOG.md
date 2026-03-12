# Changes in DataSciBench Fork

This document describes all changes made to the [DataSciBench](https://github.com/THUDM/DataSciBench) fork.

---

## 1. Accelerating Benchmark Execution

### 1.1. Reducing Number of Runs (MAX_RUNS: 10 → 3)

The original benchmark ran each task **10 times** to calculate the Pass@1 metric. We reduced this number to **3**, which decreased the total execution time with minimal loss of statistical significance.

Changes affected:
- `experiments/run_examples.py` — `max_runs` parameter
- `evaluations/check_result.py` — `MAX_RUNS` variable, dynamic completion threshold
- `evaluations/check_eval_results.py` — limit of processed results
- `evaluation_results/calculate_final_metric.py` — all Pass@1 and Average CR formulas now divide by `MAX_RUNS` instead of the hardcoded 10

### 1.2. Limiting Task Attempts (Early Stopping)

A task attempts counter (`_task_attempts`) was added to `role/sci_data_interpreter.py`. If an agent tries to solve the same subtask more than **5 times**, a `RuntimeError` is raised, and the benchmark moves to the next task. Previously, an agent could get stuck in a loop with 100+ attempts, spending hours on a single task.

### 1.3. Reducing Timeout / Forced Streaming

In MetaGPT (`MetaGPT/metagpt/provider/base_llm.py`), the `acompletion_text` method was rewritten to **always** use streaming mode (`_achat_completion_stream`), regardless of the passed `stream` parameter. This is critical for the cycle detector (see section 1.4).

### 1.4. Generation Cycle Detection (Repetition Loop Detection)

A key change to prevent hangs. Implemented in three providers:
- `MetaGPT/metagpt/provider/base_llm.py` — static method `_detect_repetition` and `RepetitionError` class
- `MetaGPT/metagpt/provider/openai_api.py` — detection in `_achat_completion_stream`, forced socket closure upon detection
- `MetaGPT/metagpt/provider/ollama_api.py` — similar detection for the Ollama provider

**Algorithm:** every N chunks (after 800 characters), the accumulated text is checked. If a string of length ≥ 40 characters repeats ≥ 6 times (or ≥ 15 characters repeats ≥ 15 times), generation is interrupted and a `RepetitionError` is raised.

**Related issue:** the `CheckData` action created a new LLM instance without our fixes. Fixed in `role/sci_data_interpreter.py` — context and LLM are now passed explicitly:
```python
check_data_action = CheckData(context=self.context)
check_data_action.set_llm(self.llm)
```

---

## 2. Infrastructure Changes

### 2.1. API Key Rotation

In `MetaGPT/metagpt/provider/openai_api.py`, API key rotation was added — for each request, the next key from the `config.api_keys` list is used. This allows for load distribution and bypassing rate limits.

### 2.2. VLM Service (Visual Language Model)

In `src/vlm_utils.py`:
- Replaced the API endpoint for VLM evaluation (from external `turboai.one` to local)
- Added logging of all VLM calls and responses to `evaluation_results/vlm_run_log.txt`
- Added file reading error handling

### 2.3. Matplotlib Fix

In `experiments/run_examples.py`, the line `os.environ["MPLBACKEND"] = "Agg"` was added — disabling the matplotlib GUI backend to prevent hangs during chart generation on a server without a display.

### 2.4. Run and Evaluation Automation

Helper scripts were added to the `scripts/` folder:
- `evaluate_model.sh` — full model evaluation pipeline (evaluate → calculate_final_metric).
- `extract_time_costs.py` — script for extracting and visualizing time costs from logs.
- `single_task/` — folder with scripts for running and evaluating individual tasks (individually or in small groups).

### 2.5. Path and Environment Fixes

- `evaluation_results/calculate_final_metric.py` — replaced hardcoded absolute path with relative `evaluation_results/results/`
- `src/logs.py` — added `os.makedirs("logs", exist_ok=True)` for automatic log directory creation
- `experiments/run_examples.py` — added environment diagnostics (logging CWD, files, `data_source_type`)
- Fixed prompt formation logic (`SPECIFY_PATH_PROMPT` is now always added)
- Fixed error with `re.escape` for model names in `evaluations/check_result.py`

### 2.6. Metric Calculation Fixes

- In `calculate_final_metric.py`, the DL Pass@1 formula was fixed — previously binary counting was used (`== "Success"`), now normalized counting is used like for other categories
- Added NaN value handling in VLM metric
- Added float value handling in VLM results

### 2.7. Disabling Token Cost Calculation

- All functions responsible for calculating token costs were commented out in `MetaGPT/metagpt/utils/cost_manager.py`, as this information is not required for our lab's tasks.

### 2.8. Miscellaneous

- Added `.gitignore`
- Removed venv from the repository
- Updated `requirements.txt`
- Merged data from this repository and Hugging Face.

---

## 3. Benchmark Generator — Auto-generation of New Examples

A prototype of the `benchmark_generator/` module was created for automatically generating new benchmark tasks using an LLM based on an arbitrary codebase. Details: [benchmark_generator/README.md](benchmark_generator/README.md).

---

## 4. Results: gemma-3-27b-it Compared to Other Models

Results were obtained on the standard DataSciBench set (55 tasks: 25 human, 20 csv_excel, 10 dl) with 3 runs per task.

| Model | Pass@1 | Avg CR | VLM |
|---|---|---|---|
| **gpt-4o-2024-05-13** | **19.82** | **17.89** | 2.10 |
| gpt-4-turbo | 17.27 | 17.36 | 1.85 |
| **gemma-3-27b-it** 🔥 | **15.64** | **14.48** | 1.61 |
| o1-mini | 13.45 | 15.43 | 1.75 |
| gpt-4o-mini | 12.73 | 17.35 | 1.65 |
| deepseek-coder-33b-instruct | 12.55 | 13.53 | 1.73 |
| deepseek-coder-6.7b-instruct | 12.55 | 13.56 | 1.30 |
| Qwen2.5-7B-Instruct | 11.64 | 10.11 | 1.44 |
| glm-4-9b-chat | 10.55 | 9.96 | 1.56 |
| Meta-Llama-3.1-8B-Instruct | 10.00 | 7.72 | 1.55 |
| glm-4-flash | 9.82 | 7.43 | 1.51 |
| claude-3-5-sonnet-20240620 | 8.00 | 11.12 | 1.44 |
| Qwen2-7B-Instruct | 6.91 | 5.90 | 1.68 |
| Yi-1.5-9B-Chat-16K | 6.18 | 4.25 | 1.82 |
| Qwen2.5-Coder-7B-Instruct | 6.18 | 7.87 | 1.35 |
| Qwen2.5-Coder-1.5B-Instruct | 6.18 | 7.52 | 1.11 |
| gemma-2-9b-it | 5.64 | 5.51 | 1.63 |

- Takes **3rd place** in Pass@1 among all models and **1st place** among open-source models

### Models in Testing Process

| Model | Pass@1 | Avg CR | VLM |
|---|---|---|---|
| Qwen3-30B-A3B | 12.73 | 13.93 | 1.57 |
| Qwen3.5 | — | — | — |

---

## 5. TODO

- [ ] **Clean dataset of errors** — incorrect paths, imports, etc. Ensure each error is a bug of the model being tested, not the benchmark itself
- [ ] **Create new examples addition pipeline** — based on `benchmark_generator`
- [ ] **Test hypotheses** — modify agent behavior code to improve benchmark performance
- [ ] **Add LLM-as-Judge** — new verification function for open-ended answers
- [ ] **Add professional Python development tools** — ruff, mypy, uv, and others
