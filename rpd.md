# Рабочий Проектный Документ (RPD) — DataSciBench

Документ предназначен для полного и исчерпывающего понимания архитектуры, файловой структуры, фактического назначения компонентов, точек входа и рабочих процессов проекта **DataSciBench**.

---

## 1. Суть проекта и архитектурный обзор

**DataSciBench** — это форк бенчмарка для языковых моделей (LLM) и автономных AI-агентов, ориентированный на решение задач в области Data Science и Machine Learning (предобработка данных, разведочный анализ, построение предиктивных моделей, обучение нейросетей, визуализация).

Проект состоит из **четырех взаимосвязанных подсистем**:
1. **Среда выполнения и оценки бенчмарка (Benchmark Harness)**:
   - **Генерация решений (Run)**: основной легковесный раннер `simple_runner` (генерация полного Python-скрипта через OpenAI-совместимый API и исполнение через изолированный `subprocess.run` с циклом исправления ошибок до 5 попыток) либо legacy-движок на базе локального MetaGPT (`SciDataInterpreter`), исполняющий сгенерированный код в песочнице Jupyter-ноутбука (`ExecuteNbCode`).
   - **Валидация артефактов (Eval)**: оценка полученных файлов и результатов по графу задач (`LinearizedDAG` на базе `metric/{task_id}/metric.yaml` и `data/{task_id}/gt/`) с расчетом Completion Rate (CR) и Pass@1.
2. **Модуль генерации новых задач (`benchmark_generator`)**: автономный подпроект автоматического создания новых DS-задач с эталонными решениями (`gt/`) и метриками (`metric.yaml`) на базе LLM (Gemini) и инструментов Model Context Protocol (MCP).
3. **Модуль дообучения (`sft`)**: пайплайн создания обучающего датасета на базе генератора задач (сплит 80/20) и fine-tuning моделей через `trl.SFTTrainer` и HuggingFace `transformers`.
4. **Система анализа логов и ошибок (`log_analysis` + `log_archive`)**: инфраструктура полуавтоматического разбора неудачных запусков моделей (CR < 1) с генерацией структурированных заданий и отчетов для ИИ-субагентов.

---

## 2. Полная карта файловой структуры

```text
DataSciBench/
├── benchmark_generator/       # Автономный модуль генерации новых DS-задач
├── data/                      # 55 задач бенчмарка (входные данные, prompt.json, gt/) + prompt CSVs
├── metric/                    # 55 папок с метриками (metric.yaml) + CSV-метрики — изолированы от data/
├── metagpt/                   # Локальное ядро MetaGPT (actions, provider, environment, utils)
├── role/                      # SciDataInterpreter (кастомная роль MetaGPT с Early Stopping)
├── src/                       # Ядро логики бенчмарка (схемы DAG, эвалюаторы, логирование, утилиты)
├── utils/                     # Строковые шаблоны Python-кода для динамического exec() в TMC/BCB
├── experiments/               # Основные точки входа (run_examples, evaluate, evaluate_tmc, simple_runner)
├── evaluations/               # bash-скрипты генерации старых моделей (32 шт.) и утилиты верификации
├── evaluation_results/        # CSV-результаты прогонов моделей (32 файла в results/) и расчет финальных метрик
├── log_analysis/              # Скрипты генерации заданий субагентам, отчеты по ошибкам и архивы выполненного
├── log_archive/               # Логи прогонов моделей (сырые .log файлы для log_analysis)
├── logs/                      # Текстовые файлы замеров времени выполнения (*_time.txt, датированные логи)
├── notebooks/                 # Jupyter-ноутбуки предобработки промптов и метрик (5 шт.)
├── notes/                     # Закрытые рабочие материалы (руководства, TeX, ключи, конфиги, table history/)
├── scripts/                   # Актуальные скрипты запусков, оценки, анализа логов и валидатор бенчмарка (validate_benchmark.py)
├── sft/                       # Скрипты подготовки SFT-датасета и дообучения
├── tests/                     # Unit-тесты компонентов бенчмарка и валидатора (test_validate_benchmark.py)
├── .github/workflows/         # Проверки в CI: ci.yaml (lint, type-check, test, validate-benchmark, changes, smoke) и reusable.yaml
├── .vscode/settings.json      # Настройки редактора: ruff как форматтер Python, format on save, pytest
├── Makefile                   # Единая точка входа для проверок (make check / lint / fix / typing / test / validate)
├── AGENTS.md                  # Контракт для ИИ-агентов, работающих с репозиторием
├── .env.example               # Шаблон переменных окружения генератора задач (OPENAI_API_KEY)
├── config2.yaml               # Образец конфига LLM-провайдера (Ollama / deepseek-coder)
├── qwen3_8b.yaml              # Пример локального конфига LLM (Ollama / qwen3:8b)
├── pyproject.toml             # Конфигурация зависимостей uv, dev-группы, ruff, mypy и pytest (Python >=3.12,<3.13)
├── uv.lock                    # Lock-файл пакетов (Python 3.12)
├── README.md                  # Публичное описание проекта
├── CHANGELOG.md               # Журнал изменений и оптимизаций форка
└── rpd.md                     # Данный документ архитектуры проекта
```

---

## 3. Детальное описание директорий и модулей

### `/metagpt` — Локальное ядро движка агентов
Фреймворк MetaGPT переведен из внешнего пакета в локальный модуль корня проекта (`import metagpt`). Из него удалены все неиспользуемые компоненты.
* `metagpt/actions/di/`:
  * `execute_nb_code.py` (`ExecuteNbCode`) — исполнение сгенерированного кода в Jupyter Kernel (`nbclient`). Если кернел `python3` отсутствует в системе, автоматически регистрирует текущее окружение (`python -m ipykernel install --user --name python3`).
  * `write_analysis_code.py` (`WriteAnalysisCode`) — генерация Python-кода шага задачи.
  * `ask_review.py` — подтверждение планов.
  * `write_plan.py` — генерация плана решения.
* `metagpt/provider/`:
  * `openai_api.py`, `ollama_api.py`, `base_llm.py` и др. — адаптеры LLM-провайдеров.
  * Включают встроенную детекцию зацикливания (`_detect_repetition` в скользящем окне) и принудительный стриминг; при обнаружении повторяющихся циклов выбрасывается `RepetitionError`.
* `metagpt/const.py`:
  * Определение корневых путей проекта (`CONFIG_ROOT = ~/.metagpt`, `METAGPT_ROOT`).
* `metagpt/logs.py`:
  * Настройка вывода логов через Loguru.

### `/role` — Кастомные роли агентов
* `role/sci_data_interpreter.py` (`SciDataInterpreter`):
  * Наследуется от `Role` MetaGPT. Адаптирован под задачи DataSciBench.
  * Включает **Early Stopping** (`_task_attempts`): если агент пытается решить одну подзадачу более 5 раз, выбрасывается `RuntimeError` для предотвращения бесконечных циклов.
  * Поддерживает режимы `plan_and_act` и `react`.
  * Собирает `plan_list`, `cost_list`, `error_counter_list` для последующей оценки через `get_results_for_eval()`.

### `/src` — Ядро логики бенчмарка
* `src/schemas/`:
  * `dag.py` — классы `Node` и `LinearizedDAG` для построения графа подзадач и их последовательной проверки (включая режим принудительной оценки `force_mode`).
  * `schemas.py` — базовые классы `Evaluator`, `Task`, `Metric`, `Rule`, `TestFunction`, `DAG`, `SciAgentBenchOutput`.
* `src/evaluator/`:
  * `cr_evaluator.py` (`CREvaluator`) — расчет Completion Rate (CR) по выполненным шагам плана.
  * `generic_evaluator.py` (`GenericEvaluator`) — базовый эвалюатор для проверки ground truth.
  * `within_range_evaluator.py` (`WithinRangeEvaluator`) — проверка численных метрик в допустимом диапазоне.
  * `larger_than_evaluator.py` (`LargerthanEvaluator`) — проверка превышения пороговых значений.
  * `evaluator_dict.py` — маппинг типов задач на классы эвалюаторов (`TM_2_EVALUATOR`: `CR` -> `CREvaluator`, `test_bool` -> `WithinRangeEvaluator`, `test_int` -> `LargerthanEvaluator`).
  * `surface_form_evaluator.py` — стаб для строковых проверок.
* `src/logs.py`:
  * `create_logger(task_id, sub_idx, ...)` — создание изолированных файловых логгеров в `data/{task_id}/{model_name}_{sub_idx}/logs.txt` и логгера времени в `logs/{model_name}_{sub_idx}_time.txt`.
  * `get_model_name(config_name)` — извлечение имени модели из YAML-конфига в `~/.metagpt/`.
* `src/utils.py`:
  * `change_dir(path)` — контекст-менеджер временной смены рабочей директории.
  * `change_metalog_path(logger, file_path)` — контекст-менеджер перенаправления логов.
  * `create_function_from_string(func_str)` — динамическая компиляция строковой Python-функции через `exec()`.
  * `load_yaml(file_path)` — безопасная загрузка YAML.
  * `list_to_dict_by_task(data_list)` — преобразование списка TMC в словарь по ключу `metric`.
  * `json_data_to_yaml(json_data, yaml_file_path)` — сохранение JSON-данных в YAML.
* `src/llm_utils.py`, `src/vlm_utils.py`, `src/vlm_config.py`:
  * `vlm_vis_quality` — оценка сгенерированных графиков по 5 критериям (промпт можно переопределить третьим аргументом из `metric.yaml`). По умолчанию судит локальный headless-запуск `claude -p` (модель `sonnet`, `--effort high`), вердикты кэшируются в `evaluation_results/vlm_bridge/{key}.score`. При `VLM_JUDGE=api` работает прежний путь через OpenAI API (`gpt-4o-mini`) с ключами из `src/vlm_config.py`. Переменные окружения: `VLM_JUDGE`, `VLM_JUDGE_CLI`, `VLM_JUDGE_MODEL`, `VLM_JUDGE_EFFORT`, `VLM_BRIDGE_TIMEOUT`.
  * `llm_text_quality` — оценка текстов по 5-балльной шкале через OpenAI API (`gpt-4o-mini`).
  * Логирование запросов в `evaluation_results/vlm_run_log.txt` и `evaluation_results/llm_run_log.txt`.

### `/utils` — Строковые сниппеты для BigCodeBench (TMC Evaluation)
* `used_libs.py` — строковая переменная `import_libs`, содержащая 57 строк импортов библиотек ML/DS.
* `test_code.py` — строковые шаблоны `test_comp_func`, `test_gt_func`, `test_metric_func` для запуска проверяемого кода внутри `ThreadPoolExecutor` с таймаутом 120 секунд.
* `cr_for_bcb.py` — расчет метрики Completion Rate для кортежей вывода BigCodeBench (`evaluate_cr`).
* `json_operator.py` — чтение и запись JSON/JSONL файлов (`read_json`, `dump_json`).
* *Все эти сниппеты динамически подставляются в `exec()` внутри `experiments/evaluate_tmc.py`.*

### `/data` и `/metric` — Данные задач и правила оценки
* **`/data`** (всего 55 задач + папка `metadata`):
  * `human_0` ... `human_142` (25 задач, написанных вручную).
  * `csv_excel_0` ... `csv_excel_48` (20 задач на табличные данные).
  * `dl_0` ... `dl_31` (10 задач на глубокое обучение: `dl_0, dl_1, dl_6, dl_9, dl_10, dl_13, dl_15, dl_16, dl_30, dl_31`).
  * Корневые файлы: `human_prompt.csv`, `csv_excel_prompt.csv`, `dl_prompt.csv`.
  * Внутри каждой задачи: `prompt.json` (постановка задачи и `data_source_type`), входные файлы данных (`.csv`, `.xlsx`, `.npy`, `.npz` и др.), папка `gt/` (эталонные скрипты и артефакты: сохраненные веса моделей, графики).
* **`/metric`** (55 папок, 1:1 соответствующих задачам из `/data`):
  * Каждая папка содержит `metric.yaml` — формализованный список TMC (Task-Metric-Code) с кодами функций проверки и правилами валидации.
  * Корневые CSV: `human_prompt_metric.csv`, `csv_excel_prompt_metric.csv`, `dl_prompt_metric.csv`, `bcb_all_data.csv`, `bcb_funcs_mod.csv`.
  * **Изоляция**: `metric/` строго отделена от `data/`, так как при выполнении агент находится в `data/{task_id}/{model_dir}` и имеет доступ к `../`. Нахождение `metric.yaml` в `data/` привело бы к утечке правильных ответов тестируемой модели.

### `/experiments` — Точки входа выполнения и оценки
* **`run_examples.py`**: Главный раннер запуска задач на моделях.
  * *Режимы работы*:
    - **По умолчанию (`simple_runner`)**: запуск `main_simple()` -> генерирует единый Python-скрипт через `OpenAILLM`, исполняет его через `subprocess.run(timeout=300)` и при падении отправляет stderr обратно модели для исправления (до 5 попыток).
    - **Legacy (`--legacy`)**: запуск `main_legacy()` -> использует MetaGPT `SciDataInterpreter` с пошаговым планированием, выполнением в Jupyter Kernel (`nbclient`) и early stopping.
  * *Флаги*: `--task_id`, `--data_type` (дефолт `human`), `--max_runs` (дефолт 3), `--config` (дефолт `test_config.yaml`), `--legacy`, `--data_source_type`, `--gt_prompt`, `--continue_gen`, `--output_dir`, `--skip_bcb`, `--use_reflection`, `--hard_retry`, `--max_retry` (дефолт 3), `--use_react`.
  * *Механика*: переключает CWD в `data/{folder}/{model_name}_{sub_idx}`, передает промпт с инструкцией `All the input source data is at the '../' folder. And all the output files should be saved at the currect folder './'`. Пишет `logs.txt`, `sys_logs.txt` и сохраняет метаданные в `data/{folder}/{model_name}_outputs.jsonl`.
* **`evaluate.py`**: Главный скрипт валидации результатов.
  * Сканирует `data/`, для каждой модели читает сгенерированные файлы в `data/{task_id}/{model_run_id}`, загружает `metric/{task_id}/metric.yaml` и `data/{task_id}/gt/`.
  * Строит `LinearizedDAG` (в `force_mode`), выполняет узлы через `evaluate_node()`, рассчитывает CR через `CREvaluator` и записывает строки результатов в `evaluation_results/results/{model_name}_results.csv`.
* **`simple_runner.py`**: Легковесный раннер одиночных скриптов без планировщика MetaGPT через прямой вызов `OpenAILLM(config.llm)` и `subprocess.run(timeout=300)` с циклом повторов ошибок (до `MAX_RETRIES=5`).
* **`evaluate_tmc.py`**: Оценщик для BigCodeBench (TMC). Извлекает функцию ответа `def task_func` из плана, динамически подставляет строковые сниппеты из `/utils` (`import_libs`, `test_comp_func`, `test_gt_func`, `test_metric_func`) и исполняет их через `exec()` внутри `ThreadPoolExecutor` с таймаутом 120с.
* **`show_bcb_eval_results.py`**: Читает `data/*/{model_id}_tmc_results.jsonl` и выводит сводную статистику CR и Success Rate по задачам BCB.
* **`first-example.py`**: Тестовый запуск `SciDataInterpreter` на одном промпте.
* **`di_example.py`**: Тестовый запуск стандартного `DataInterpreter` MetaGPT.
* **`test_config.py`**: Проверка чтения конфига через `Config.from_sab_config()` и инициализации роли `SciDataInterpreter`.

### `/scripts` — Актуальные скрипты автоматизации и анализа
* **`evaluate_model.sh [MODEL_NAME]`**: Запускает `python -m experiments.evaluate --task_id all --model_id "$MODEL_NAME"`; без аргумента берётся `Qwen3-30B-A3B`.
* **`run_gemma3.sh`**, **`run_qwen3.5_27b.sh`**, **`run_qwen3_30b.sh`**: Циклически запускают `python -m experiments.run_examples --data_type $data_type --max_runs 3 --config <config.yaml>` по категориям `human`, `csv_excel`, `dl`.
* **`run_kimi.sh`**: Аналогичный запуск с `--max_runs 10` и конфигом `kimi.yaml`.
* **`single_task/`**:
  * `run_single_task.sh`: запуск генерации одной задачи (`TASK_ID="code_gen_001"`, `--max_runs 1`, сброшенные фильтры `data_source_type` и `data_type`).
  * `evaluate_single_task.sh`: запуск `python -m experiments.evaluate --task_id "$TASK_ID" --model_id "$MODEL_NAME"`; `TASK_ID` (`human_3`) и `MODEL_NAME` (`gemma-3-27b-it`) зашиты в скрипт, аргументов он не принимает.
  * `calculate_metric_single_task.sh`: запуск `python -m evaluation_results.calculate_final_metric`.
  * `run_batch_tasks.sh`: пакетный запуск генерации для массива задач `TASKS=("dl_9" ...)`.
* **`extract_time_costs.py`**: Сканирует `data/`, извлекает `time_cost` из `SciAgentBenchOutput` (только файлы `Qwen3-30B-A3B_outputs.jsonl`), сортирует по времени, строит скаттерплот с помощью `matplotlib` и сохраняет график в `time_costs_scatter.png`.
* **`parse_logs.py`**: Парсит `log_archive/Qwen3.5-27B.log` и `log_archive/context/Qwen3.5-27B.log` через регулярные выражения, подсчитывает `SyntaxError`, `IndentationError`, `NameError`, `AttributeError`, количество отсечек по длине контекста (`finish_reason=length`) и частоту ошибок.
* **`check_runs.py`**: Сканирует папки отчетов `log_analysis/reports` и `reports_missing`, проверяет наличие идеальных прогонов (`CR == 1.0`) по таблице результатов и сохраняет сводку в `cr_check_results.csv`.
* **`check_missing_results.py`**: Проверяет таблицу результатов на наличие всех ожидаемых ранов (10 для gemma, 3 для qwen) для каждой задачи из `data/`.
* **`analyze_logs.py`**: Парсер логов, классифицирующий сбои по типам исключений и формирующий `analysis_data.json`.
* **`validate_benchmark.py`**: Валидатор целостности бенчмарка (`python -m scripts.validate_benchmark`, флаги `--task-id` (дефолт `all`), `--root` (дефолт `.`), `--json`). Возвращает код 1, если найдена хотя бы одна проблема или не проверено ни одной задачи. Текущее состояние: проверено 55 задач, 0 проблем. Проверки:
  * `root` — наличие каталогов `data/` и `metric/` в корне.
  * `layout` — соответствие 1:1 между `data/{task_id}` и `metric/{task_id}` (каталог `metadata` из проверки исключён).
  * `prompt` — наличие `data/{task_id}/prompt.json`, что это JSON-объект с ключами `prompt` и `data_source_type`, а `prompt` — непустая строка.
  * `gt` — наличие непустой папки `data/{task_id}/gt/`.
  * `metric-file` / `metric-schema` — `metric.yaml` существует, парсится и содержит непустой список `TMC-list`; каждый его элемент — словарь с ключами `code`, `function`, `metric`, `task_name`.
  * `metric-name` — имя метрики является непустой строкой и не равно зарезервированному `CR` (иначе оценка получит `CREvaluator` без тестовой функции и упадёт).
  * `metric-duplicate` — повторяющиеся имена метрик внутри задачи (при оценке дубликат теряется).
  * `code` — код метрики **разбирается через `ast.parse`, а не исполняется**. Проверяется: код компилируется; первым callable-объектом верхнего уровня (именно он уйдёт в оценку) объявлена функция через `def`, а не класс, импортированное имя или присваивание; функция вызывается ровно одним позиционным аргументом; функция не объявлена через `async def` (иначе оценка получит корутину, и метрика окажется истинной при любом результате); в теле нет обращений к именам, которых не будет в области видимости функции — их ищет `symtable`, потому что `create_function_from_string` исполняет код с раздельными словарями globals и locals, и константа верхнего уровня даёт `NameError` при вызове метрики.
  * `ground-truth` — путь из ключа `ground_truth` указывает на существующий файл в `data/{task_id}/gt/` и не выходит за пределы `data/{task_id}/`.
  * `evaluator` — имя метрики не должно маппиться в `TM_2_EVALUATOR` на эвалюатор, отличный от `CREvaluator`: `WithinRangeEvaluator` и `LargerthanEvaluator` требуют аргумент `output`, которого `Node.evaluate_node` не передаёт, поэтому такая запись даёт ноль независимо от ключа `rule`.

### `/evaluations` — Скрипты генерации старых моделей и утилиты верификации
* **`bash/*.sh` (32 файла)**: Шелл-скрипты генерации решений (`python -m experiments.run_examples --data_type $data_type --max_run 10 --config $config`) для исторических моделей (CodeLlama, Llama-3, Qwen2, Claude-3, GPT-4o, StarCoder2 и др.).
* **`evaluate_all.sh`**: Шелл-скрипт оценки — итерируется по списку моделей и вызывает `python -m experiments.evaluate --task_id all --model_id $model`.
* **`check_result.py`**: Скрипт проверки прогресса генерации: проверяет наличие и размер `sys_logs.txt` (>100 B) и `logs.txt` (>10 B) в `data/{task_id}/{model_name}_{sub_idx}` и выводит число завершенных задач по типам (`bcb` /167, `dl` /10, `human` /25, `csv` /20).
* **`check_eval_results.py`**: Интерактивный терминальный инструмент проверки результатов BigCodeBench: построчно выводит промпт, ответ модели и CR из `data/{dir}/{model_id}_tmc_results.jsonl` и запрашивает ручное подтверждение пользователя `(y/n)`.

### `/evaluation_results` — Результаты и финальный расчет
* **`results/`**: 32 CSV-файла с результатами оценки отдельных моделей (`{model_name}_results.csv`). Колонки: `model_name`, `run_id`, `data_name`, `task_name`, `metric_name`, `function_name`, `result_value`, `result_cr`, `result_type`.
* **`calculate_final_metric.py` / `.ipynb`**:
  * Читает CSV-файлы моделей из `evaluation_results/results/`.
  * Рассчитывает Pass@1 (эмпирическое матожидание: каждый ран с `CR == 1.0` дает вклад 1/3 в счетчик задачи), средний CR (общий по 169 функциям, Human по 83, CSV по 59, DL по 27), баллы VLM/LLM и средний CR для Top-5 ключевых функций (`Data Quality Score`, `Plot Validity`, `Data Accuracy`, `Visualization Completeness`, `Model Accuracy`).
  * Константы: `TOTAL_DATA_NUM = 55`, `HUMAN_DATA_NUM = 25`, `CSV_DATA_NUM = 20`, `DL_DATA_NUM = 10`.
  * Дописывает итоговую строку метрик в `evaluation_results/results/final_results.csv`.
* **`top_passed_tasks.py`**: Парсит `combined_results.csv` и ранжирует задачи по частоте успешного прохождения моделями с `CR == 1.0`.
* **`combined_results.csv`**, **`final_results.csv`**, **`final_results_orig.csv`**, **`top.txt`**: Сводные таблицы и файлы результатов.

### `/benchmark_generator` — Модуль автогенерации задач
* **`scripts/run_standard.sh`**: Устанавливает модель `gemini-3-flash-preview` и базовый URL Google AI Studio, после чего запускает `pipeline_standard.py`. Ключ в скрипте не хранится — он читается из `.env` в корне проекта.
* **`scripts/run_code_mode.sh`**: Аналогичный запуск пайплайна с поддержкой MCP-серверов (`pipeline_code_mode.py`).
* Секретов оба скрипта не содержат и хранятся в репозитории: ключ `OPENAI_API_KEY` подхватывается из `.env` через `Settings` (`benchmark_generator/core/config.py`), а сам `.env` исключён из git.
* **`core/`**: `healthcheck.py`, `llm.py`, `config.py`, `models.py`, `mcp_tools.py`.
* **`tools/generate_tasks.py`**: Генерирует папки задач с `prompt.md` и входными данными по кодовой базе (`code_base.txt`) и теме (`topic.md`).
* **`tools/solve_task.py`**: Генерирует `solution.py` и `metrics.py`, исполняет решение в подпапке `gt/` задачи и создает эталонные артефакты с проверкой в `verify_log.txt`.
* **`tools/pack_task.py`**: Переносит готовую задачу в основной бенчмарк:
  * `prompt.md` -> `data/{task_id}/prompt.json`
  * Входные файлы данных -> `data/{task_id}/`
  * `gt/` -> `data/{task_id}/gt/`
  * `metrics.py` -> `metric/{task_id}/metric.yaml`
* **`tests/`**: 43 автономных pytest-теста (`test_mcp_tools.py`, `test_models.py`, `test_pack.py`, `test_parse.py`, `test_roundtrip.py`).

### `/sft` — Дообучение моделей
* **`create_sft_dataset.py`**: Запускает генерацию и решение задач через `benchmark_generator.tools`, верифицирует через `verify_log.txt`, сохраняет 80% задач в `sft/sft_dataset.jsonl` (в формате пар сообщений `user`/`assistant`), а 20% упаковывает через `pack_single_task` в `data/` и `metric/` с префиксом `sft_test_XXX`.
* **`train_model.py`**: Загружает `sft/sft_dataset.jsonl`, настраивает `AutoModelForCausalLM` (частичная разморозка последних `unfreeze_layers=2` слоев и `lm_head`), `DataCollatorForCompletionOnlyLM` и запускает обучение через `trl.SFTTrainer` с сохранением чекпоинтов в `sft/checkpoints/` и финальной модели в `sft/tuned_model/` *(зависимости `transformers`, `trl`, `datasets` используются в GPU-окружении кластера)*.

### `/log_analysis` и `/log_archive` — Разбор ошибок
Обе директории целиком исключены из репозитория через `.gitignore` и существуют только в локальной рабочей копии: там лежат рабочие промпты для ИИ-субагентов и сырые логи прогонов. По той же причине `log_analysis/` не входит в зону проверки mypy.
* **`log_archive/`**: Хранилище сырых логов прогонов (`Qwen3-30B-A3B.log`, `Qwen3.5-27B.log`, `квены после исправления задач.log`, `context/Qwen3.5-27B.log`, `after cleaning/gemma3_run.log`, `after cleaning/gemma3_run_dop.log`).
* **`log_analysis/generate_prompts.py`**: Читает CSV с результатами (`evaluation_results/our_models.csv`, строки с `CR < 1.0` отбираются заранее, сам скрипт их не фильтрует) и генерирует Markdown-файлы с инструкциями для субагентов в папку `prompts/`.
* **`log_analysis/generate_prompts_missing.py`**: Генератор промптов по недостающим ранам в `prompts_missing/`.
* **`log_analysis/done/`** и **`done_missing/`**: Папки, куда перемещаются обработанные промпты (137 и 43 файла соответственно).
* **`log_analysis/reports/`** и **`reports_missing/`**: Готовые аналитические отчеты субагентов по каждой ошибке с классификацией причины (ошибка модели / ошибка бенчмарка / ошибка инфраструктуры).

### `/notes` — Локальные материалы (Вне Git)
Директория добавлена в `.gitignore` и исключается из синхронизации с кластером.
* **`cluster_guide.md`**: Инструкция по подключению к кластеру `gpu10` (VPN `vpn.conf`, SSH `malyshev_sa@gpu10`, пути к Conda-окружению `/home/malyshev_sa/miniconda/envs/datascibench/`, настройка прокси `http://10.11.1.254:3128`).
* **`заметки.md`**: Команды очистки логов на кластере и параметры архивации `tar -czvf DataSciBench.tar.gz .` с исключениями.
* **`key.md`**: Локальные API-ключи и конфигурации эндпоинтов LiteLLM.
* **`план.md`**, **`промпт *.md`**: Журнал гипотез, промпты экспериментов и заметки по развитию проекта.
* **`table history/`**: История CSV-таблиц прогонов (`our_models.csv`, `our_models_orig.csv`, `Qwen3-30B-A3B_results.csv` и др.).
* **`tex/`**: `hypotheses_tz.tex` и `ТЗ на проверку гипотез.pdf`; остальные LaTeX-исходники и PDF (`presentation_final.tex`, `thezis.tex`, `project_article.pdf`, `research_proposal.pdf`) лежат прямо в `notes/`.

### `/notebooks` — Исследовательские ноутбуки
* `get_function_stats.ipynb` — расчет статистики по типам функций задач.
* `get_metric_stats.ipynb` — анализ распределения метрик.
* `preprocess_prompt.ipynb` — предобработка исходных промптов задач и раскладка по `data/{task_id}/prompt.json`.
* `preprocess_metrics.ipynb` — предобработка исходных CSV-метрик и генерация `metric/{task_id}/metric.yaml`.
* `preprocess_bcb_metric_prompt.ipynb` — препроцессинг задач BigCodeBench.

### `/tests` — Unit-тесты
Каталог является пакетом (`__init__.py`). Прогон `uv run pytest` по всему проекту даёт **115 passed, 4 skipped**: `tests/` — 72 passed и 4 skipped, `benchmark_generator/tests/` — 43 passed.
* `load_yaml_test.py` — 4 теста `unittest`: загрузка YAML, конвертация TMC-списка и динамическая компиляция функций.
* `json_yaml_test.py` — тест сериализации `json_data_to_yaml`.
* `test_validate_benchmark.py` — 66 тестов валидатора целостности бенчмарка `scripts/validate_benchmark.py`.
* `test_example.py` — базовый `unittest.TestCase` для проверки тестового DAG; оба теста помечены `@pytest.mark.skip`, потому что ничего не проверяют (пустое тело и сравнение константы 0 с 10).
* `str_to_func_test.py` — тест динамической компиляции функции из строки (`create_function_from_string`).
* `test_context_manager.py` — легаси-тесты контекст-менеджеров `change_dir` и `change_metalog_path`, помечены `@pytest.mark.skip` с объяснением: написаны под несуществующий API (`change_metalog_path` ожидает loguru-логгер с `remove`/`add` и атрибутом `logfile`, тогда как `src.logs.create_logger` отдаёт `logging.Logger`; пути `/path/to/new/dir` не существуют). Вместе с двумя заглушками из `test_example.py` это и есть 4 skipped.
* `schema_test.py` — ручной скрипт под `if __name__ == "__main__"`: инициализация `LinearizedDAG` и расчёт CR через `CREvaluator`; тестов pytest из него не собирает.
* `testcases/` — YAML-фикстуры (`example.yaml`); `test_outputs/` — каталог для файлов, создаваемых тестами.

---

## 4. Жизненный цикл задачи (Runtime & Sandboxing)

```text
[Генерация решения (Run)]
experiments.run_examples
  ├── Чтение data/{task_id}/prompt.json
  ├── Инициализация логгеров в data/{task_id}/{model}_{run}/logs.txt и sys_logs.txt
  ├── os.chdir("data/{task_id}/{model}_{run}")
  │
  ├── Вариант А: Simple Runner (default)
  │   ├── OpenAILLM генерирует полный скрипт решения
  │   ├── Запуск через subprocess.run(sys.executable, "-c", code, timeout=300)
  │   └── При ошибке: передача stderr обратно LLM (до 5 попыток)
  │
  └── Вариант Б: Legacy MetaGPT (--legacy)
      ├── Исполнение кода в Jupyter Kernel через ExecuteNbCode (nbclient + ipykernel)
      ├── Пошаговое планирование (WritePlan -> WriteAnalysisCode)
      └── Early Stopping (RuntimeError при >5 попытках на одну подзадачу)
  │
  └── Запись метаданных в data/{task_id}/{model}_outputs.jsonl

[Валидация артефактов (Eval)]
experiments.evaluate
  ├── Сканирование data/{task_id}/{model}_{run}
  ├── Чтение metric/{task_id}/metric.yaml -> построение LinearizedDAG (force_mode=True)
  ├── Чтение эталонов из data/{task_id}/gt/
  ├── os.chdir("data/{task_id}/{model}_{run}")
  ├── Выполнение функций проверки узлов через create_function_from_string()
  ├── При необходимости: VLM/LLM оценка (vlm_vis_quality / llm_text_quality)
  ├── Расчет Completion Rate (CREvaluator.calcualte_cr)
  └── Запись в evaluation_results/results/{model}_results.csv

[Финальная агрегация]
evaluation_results.calculate_final_metric
  ├── Чтение evaluation_results/results/{model}_results.csv
  ├── Расчет Pass@1 (доля ранов с CR == 1.0 от общего числа попыток), среднего CR (169 функций), VLM/LLM score, Top-5 CR
  └── Дозапись строки в evaluation_results/results/final_results.csv
```

---

## 5. Справочник команд запуска (из корня DataSciBench)

### Запуск экспериментов (генерация решений):
```bash
# <CONFIG_NAME>.yaml — имя файла в ~/.metagpt/; имя config2.yaml запрещено assert-ом в src/logs.py

# Все задачи (по 3 прогона на задачу через simple_runner)
python -m experiments.run_examples --max_runs 3 --config <CONFIG_NAME>.yaml

# Конкретная категория (human / csv_excel / dl)
python -m experiments.run_examples --data_type human --max_runs 3 --config <CONFIG_NAME>.yaml

# Запуск в legacy-режиме MetaGPT (SciDataInterpreter)
python -m experiments.run_examples --data_type human --max_runs 3 --config <CONFIG_NAME>.yaml --legacy

# Одиночная задача через скрипт
./scripts/single_task/run_single_task.sh
```

### Валидация и расчет метрик:
```bash
# Оценка всех задач модели
python -m experiments.evaluate --task_id all --model_id <MODEL_NAME>

# Оценка одной задачи (TASK_ID и MODEL_NAME правятся внутри скрипта, аргументов он не принимает)
./scripts/single_task/evaluate_single_task.sh

# Расчет Pass@1 и CR
python -m evaluation_results.calculate_final_metric

# Полный цикл оценки модели
./scripts/evaluate_model.sh <MODEL_NAME>
```

### Генерация задач и SFT:
```bash
# Автогенерация задач
./benchmark_generator/scripts/run_standard.sh
./benchmark_generator/scripts/run_code_mode.sh

# Сборка SFT датасета и обучение
python -m sft.create_sft_dataset --codebase ./context.txt --count 10
python -m sft.train_model --model_id google/gemma-2-2b-it --epochs 3
```

### Тестирование:
```bash
# Все тесты проекта (115 passed, 4 skipped)
uv run pytest

# Только тесты ядра бенчмарка (72 passed, 4 skipped)
uv run pytest tests/

# Только тесты benchmark_generator (43 теста, все проходят успешно)
uv run pytest benchmark_generator/tests/
```

### Проверки качества (формат, линт, типы, тесты, валидация):
```bash
# Все проверки разом — цель по умолчанию; ничего не меняет в рабочем дереве, ровно то же идет в CI
make check

# Установка зависимостей вместе с dev-группой
make install    # uv sync --all-groups

# По отдельности
make lint       # ruff format --check + ruff check
make fix        # ruff format + ruff check --fix (переписывает файлы)
make typing     # mypy
make test       # pytest
make validate   # python -m scripts.validate_benchmark
# Все цели вызывают инструменты как `uv run --no-sync --frozen`, то есть тем же способом, что CI,
# и не трогают ни окружение, ни uv.lock. На свежем клоне сначала нужен `make install`.

# Валидатор целостности бенчмарка напрямую
python -m scripts.validate_benchmark
python -m scripts.validate_benchmark --task-id human_3
python -m scripts.validate_benchmark --json
```

---

## 6. Инструменты качества кода и CI

Единая точка входа для всех проверок — `Makefile`, контракт для ИИ-агентов зафиксирован в `AGENTS.md`. `pyproject.toml` — единственный источник конфигурации ruff, mypy и pytest, включая версии самих инструментов.

* **Ruff** (`[tool.ruff]`) — форматтер и линтер: `line-length = 120`, `indent-width = 4`, `target-version = "py312"`. Включён `select = ["ALL"]` с чёрным списком в `ignore`: `COM812` и `ISC001` (конфликтуют с форматтером), `D` (докстринги в проекте не пишутся), `S102`, `S307`, `S603`, `S108` (`exec`, `eval` и `subprocess` — штатная механика запуска решений и проверок метрик; `S101` глобально не отключён, assert разрешён точечно через `per-file-ignores`), `T20` (`print` — штатный вывод скриптов), `ANN` (аннотации требует mypy в строгой зоне), `ERA001`, `RUF001` и `RUF003` (кириллица в строках и комментариях), `PTH`, `INP001`, `TID252`, `FBT`, `TD`, `FIX`, `EM`, `TRY003`, `E501`, `PLR2004`, `BLE001`, `PLC0415`, `N806`, `C901`, `PLR0912`, `PLR0915`. Из проверки исключены (`extend-exclude`) `metagpt/`, `data/`, `logs/`, `log_archive/`, `notebooks/`. Точечные послабления заданы в `[tool.ruff.lint.per-file-ignores]` для `tests/`, `benchmark_generator/tests/`, ноутбуков, `experiments/`, `evaluations/`, реэкспортов в `__init__.py` и отдельных модулей (`S101` — для `src/logs.py`, `src/schemas/dag.py`, `scripts/validate_benchmark.py`, `experiments/evaluate.py`).
* **Mypy** (`[tool.mypy]`) — градиент строгости: `python_version = "3.12"`, плагин `pydantic.mypy`, список проверяемых директорий задан в `files` (`src`, `benchmark_generator`, `sft`, `experiments`, `scripts`, `utils`, `role`, `tests`, `evaluation_results`, `evaluations`, `log_analysis`), поэтому `mypy` запускается без аргументов.
  * **Строгая зона** — `src.*`, `benchmark_generator.*`, `sft.*`: `disallow_untyped_defs`, `disallow_incomplete_defs`, `warn_return_any`, то есть аннотации функций обязательны.
  * **Базовая зона** — остальной свой код: глобальный `check_untyped_defs = true` плюс `warn_redundant_casts`, `warn_unused_ignores`, `warn_unreachable`, `local_partial_types`.
  * **Исключения** — вендоренный `metagpt.*` (`ignore_errors = true`) и данные `data/` (в `files` не входят). Для сторонних библиотек без стабов проставлен `ignore_missing_imports`.
* **Pytest** (`[tool.pytest.ini_options]`): `testpaths = ["tests", "benchmark_generator/tests"]`, ретраи `--reruns 2 --reruns-delay 5` включены **только на сетевых исключениях** (`--only-rerun` для `ReadTimeout`, `ConnectError`, `ConnectTimeout`, `APITimeoutError`, `APIConnectionError`) — падение по логике теста не перезапускается.
* **Dev-инструменты** вынесены в `[dependency-groups] dev` (`mypy`, `ruff`, `pytest`, `pytest-rerunfailures`, стабы `pandas-stubs`, `types-pyyaml`) и ставятся через `uv sync --all-groups`.
* **`.gitignore`**: игнорируются точечно логи и артефакты прогонов (`/logs/`, `/log_archive/`, `*.log`, `logs.txt`, `sys_logs.txt` — с парными исключениями `!data/*/gt/logs.txt` и `!data/*/gt/sys_logs.txt`, потому что эталонные логи задач хранятся в репозитории, — весь каталог `log_analysis/`), кеши (`.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`), `notes`, `log_analysis/`, `.env`, `src/vlm_config.py` (два последних содержат секреты) и артефакты оценки `evaluation_results/results/gt_results.csv`, `evaluation_results/vlm_bridge/`, `vlm_run_log.txt`, `llm_run_log.txt`; `.vscode/` исключён целиком, кроме `.vscode/settings.json`. Прежние широкие маски `log*` и `scripts` убраны — они прятали от git и от линтера рабочий код: `src/logs.py`, весь каталог `scripts/` и `metagpt/logs.py`.
* **CI** (`.github/workflows/ci.yaml`): на каждый push и pull request идут джобы `lint`, `type-check`, `test`, `validate-benchmark`. Последние три вызывают `.github/workflows/reusable.yaml` (`uv sync --locked --all-groups`, затем `uv run --no-sync <test_command>`), `lint` ставит только группу `dev` (`uv sync --locked --only-group dev`), поэтому локальный `make lint` и CI берут одну и ту же версию ruff. Джоба `changes` определяет, затронуты ли изменениями `data/`, `metric/`, `src/evaluator/`, `src/schemas/`, `src/utils.py`, `src/vlm_utils.py`, `experiments/evaluate.py`, `pyproject.toml` или `uv.lock`, и при попадании запускает джобу `smoke`: `python -m experiments.evaluate --task_id <smoke_task_id> --model_id gt` оценивает эталонное решение из `data/{task_id}/gt/` его же метриками и падает, если Completion Rate не равен 1.0. Список задач для прогона выбирает `.github/scripts/select_smoke_tasks.sh` по изменённым путям `data/{task_id}/` и `metric/{task_id}/`, отбрасывая задачи с метриками, требующими LLM- или VLM-судью, и обрезая список до `MAX_SMOKE_TASKS = 3`; если не выбрано ничего, берётся `human_3` — задача без VLM-функций, прогон не требует ключей API. Вручную джоба запускается через `workflow_dispatch` с параметром `smoke_task_id`.

---

## 7. Конфигурация и окружение

* **Python**: `>=3.12,<3.13` (управляется через `uv`, lock-файл `uv.lock`).
* **Конфиги моделей MetaGPT / OpenAI API**: читаются только из `~/.metagpt/{config_name}.yaml` (`Config.from_home` и `src.logs.get_model_name`), имя файла задаётся флагом `--config` (дефолт `test_config.yaml`); имя `config2.yaml` запрещено assert-ом в `src/logs.py`. Корневые `config2.yaml` и `qwen3_8b.yaml` — образцы, которые нужно скопировать в `~/.metagpt/`.
  Структура конфигурации:
  ```yaml
  llm:
    api_type: "openai" # или "ollama"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o"
    api_key: "sk-..."
  ```
* **Ключ генератора задач**: `benchmark_generator/core/config.py` описывает настройки классом `Settings` на базе `pydantic_settings.BaseSettings` с `env_prefix = "BENCH_GEN_"` и `env_file = ".env"`. Ключ провайдера читается из переменной `OPENAI_API_KEY` — сначала из окружения, затем из `.env` в корне проекта; шаблон `.env.example` показывает, какие переменные нужны. Модель и адрес провайдера (`BENCH_GEN_LLM_MODEL`, `BENCH_GEN_LLM_API_BASE`) секретами не являются и заданы прямо в скриптах запуска.
* **Конфигурация VLM/LLM оценки**: `src/vlm_config.py` задает `API_KEY` и `BASE_URL` для текстовой оценки и для судьи по изображениям в режиме `VLM_JUDGE=api`.
* **Удаленный кластер (GPU)**:
  * Эксперименты запускаются на удаленном сервере (например, `malyshev_sa@gpu10`).
  * Синхронизация данных выполняется через `rsync -avz --delete data/ malyshev_sa@gpu10:~/DataSciBench/data/`.
  * Локальные секреты, логи и архивы исключаются из передачи через `.gitignore` и `--exclude` в `tar`.

---

## 8. Ключевые особенности и механизмы защиты

* **Dual-Engine Execution**: Поддержка быстрого скриптового раннера `simple_runner` (с циклом исправления ошибок) и полноценного агентного фреймворка `SciDataInterpreter` (`--legacy`).
* **Early Stopping**: В режиме MetaGPT агенты ограничены максимальным числом попыток на одну подзадачу (`_task_attempts` <= 5), чтобы избежать бесконечных циклов "ошибка-исправление".
* **Repetition Detection**: Встроенный в адаптеры LLM механизм детекции зацикливания генерации с немедленным прерыванием по `RepetitionError`.
* **Forced Streaming**: Все запросы к LLM используют стриминг для работы детектора циклов в реальном времени.
* **Environment Diagnostics**: При запуске в `sys_logs.txt` логируются CWD, список файлов в CWD, состав родительской директории и `data_source_type`.
* **Изоляция данных и метрик**: Физическое разделение `data/` и `metric/` гарантирует отсутствие утечки эталонов при выполнении сгенерированного кода в песочнице.
