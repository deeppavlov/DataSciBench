from __future__ import annotations

import argparse
import ast
import builtins
import json
import os
import symtable
import sys
from dataclasses import asdict, dataclass

import src.utils
from src.evaluator.cr_evaluator import CREvaluator
from src.evaluator.evaluator_dict import TM_2_EVALUATOR
from src.utils import load_yaml

DATA_DIR = "data"
METRIC_DIR = "metric"
ROOT_ID = "-"
NON_TASK_DIRS = frozenset({"metadata"})
PROMPT_KEYS = frozenset({"prompt", "data_source_type"})
TMC_KEYS = frozenset({"code", "function", "metric", "task_name"})
RESERVED_METRIC = "CR"
MAYBE_CALLABLE = (ast.Lambda, ast.Name, ast.Attribute, ast.Call)
CALLABLE_KINDS = frozenset({"def", "async-def", "class", "import", "assign"})
KIND_NAMES = {"class": "класс", "import": "импортированное имя", "assign": "присвоенное значение"}
RUNTIME_GLOBALS = frozenset(dir(builtins)) | frozenset(vars(src.utils))


@dataclass
class Problem:
    task_id: str
    check: str
    message: str


def list_dirs(path: str) -> set[str]:
    if not os.path.isdir(path):
        return set()
    return {name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and name not in NON_TASK_DIRS}


def check_root(root: str) -> list[Problem]:
    return [
        Problem(ROOT_ID, "root", f"нет каталога {name}/ в {os.path.abspath(root)}")
        for name in (DATA_DIR, METRIC_DIR)
        if not os.path.isdir(os.path.join(root, name))
    ]


def check_layout(root: str) -> tuple[list[str], list[Problem]]:
    data_ids = list_dirs(os.path.join(root, DATA_DIR))
    metric_ids = list_dirs(os.path.join(root, METRIC_DIR))
    problems = [
        Problem(task_id, "layout", f"есть {DATA_DIR}/{task_id}, но нет {METRIC_DIR}/{task_id}")
        for task_id in sorted(data_ids - metric_ids)
    ]
    problems += [
        Problem(task_id, "layout", f"есть {METRIC_DIR}/{task_id}, но нет {DATA_DIR}/{task_id}")
        for task_id in sorted(metric_ids - data_ids)
    ]
    return sorted(data_ids | metric_ids), problems


def check_prompt(root: str, task_id: str) -> list[Problem]:
    path = os.path.join(root, DATA_DIR, task_id, "prompt.json")
    if not os.path.isfile(path):
        return [Problem(task_id, "prompt", "нет файла prompt.json")]
    try:
        with open(path) as file:
            payload = json.load(file)
    except Exception as error:
        return [Problem(task_id, "prompt", f"prompt.json не читается как JSON: {error}")]
    if not isinstance(payload, dict):
        return [Problem(task_id, "prompt", "prompt.json содержит не объект")]
    missing = sorted(PROMPT_KEYS - set(payload))
    if missing:
        return [Problem(task_id, "prompt", f"в prompt.json нет ключей: {', '.join(missing)}")]
    prompt = payload["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        return [Problem(task_id, "prompt", "значение prompt пусто или не является строкой")]
    return []


def check_gt(root: str, task_id: str) -> list[Problem]:
    path = os.path.join(root, DATA_DIR, task_id, "gt")
    if not os.path.isdir(path):
        return [Problem(task_id, "gt", "нет папки gt/")]
    if not os.listdir(path):
        return [Problem(task_id, "gt", "папка gt/ пуста")]
    return []


def binding_names(node: ast.stmt) -> list[tuple[str, str]]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return [(node.name, "async-def" if isinstance(node, ast.AsyncFunctionDef) else "def")]
    if isinstance(node, ast.ClassDef):
        return [(node.name, "class")]
    if isinstance(node, ast.Import):
        return [(alias.asname or alias.name.split(".")[0], "module") for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [(alias.asname or alias.name, "import") for alias in node.names]
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        kind = "assign" if isinstance(node.value, MAYBE_CALLABLE) else "value"
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return [(target.id, kind) for target in targets if isinstance(target, ast.Name)]
    return []


def top_level_bindings(tree: ast.Module) -> list[tuple[str, str, ast.stmt]]:
    found: dict[str, tuple[str, ast.stmt]] = {}
    for node in tree.body:
        for name, kind in binding_names(node):
            if name not in found:
                found[name] = (kind, node)
    return [(name, kind, node) for name, (kind, node) in found.items()]


def signature_problem(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    args = node.args
    slots = len(args.posonlyargs) + len(args.args)
    required = slots - len(args.defaults)
    required_keyword = sum(1 for default in args.kw_defaults if default is None)
    if required_keyword:
        return f"требует обязательных именованных аргументов: {required_keyword}"
    if slots == 0 and args.vararg is None:
        return "не принимает позиционных аргументов"
    if required > 1:
        return f"принимает обязательных позиционных аргументов: {required}"
    return None


def free_globals(table: symtable.SymbolTable) -> set[str]:
    names = {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_global() and not symbol.is_assigned() and symbol.get_name() not in RUNTIME_GLOBALS
    }
    for child in table.get_children():
        names |= free_globals(child)
    return names


def undefined_names(code: str, func_name: str) -> list[str]:
    try:
        module = symtable.symtable(code, "<metric>", "exec")
    except (SyntaxError, ValueError):
        return []
    for child in module.get_children():
        if child.get_name() == func_name:
            return sorted(free_globals(child))
    return []


def check_code(task_id: str, name: str, code: object) -> list[Problem]:
    if not isinstance(code, str):
        return [Problem(task_id, "code", f"метрика {name!r}: код не является строкой")]
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return [Problem(task_id, "code", f"метрика {name!r}: код не компилируется: {error}")]

    candidates = [item for item in top_level_bindings(tree) if item[1] in CALLABLE_KINDS]
    if not candidates:
        return [
            Problem(
                task_id,
                "code",
                f"метрика {name!r}: в коде нет ни одного вызываемого объекта, "
                f"в оценку уйдёт None и вызов метрики упадёт",
            )
        ]

    first_name, kind, node = candidates[0]
    if kind not in ("def", "async-def"):
        return [
            Problem(
                task_id,
                "code",
                f"метрика {name!r}: в оценку уйдёт {first_name!r} ({KIND_NAMES[kind]}), "
                f"а не функция, объявленная через def",
            )
        ]

    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    problems: list[Problem] = []
    if kind == "async-def":
        problems.append(
            Problem(
                task_id,
                "code",
                f"метрика {name!r}: функция {first_name!r} объявлена через async def, "
                f"оценка не ждёт корутину и метрика окажется истинной при любом результате",
            )
        )
    reason = signature_problem(node)
    if reason is not None:
        problems.append(
            Problem(
                task_id,
                "code",
                f"метрика {name!r}: функция {first_name!r} {reason}, "
                f"а оценка вызывает её ровно с одним позиционным аргументом",
            )
        )
    missing = undefined_names(code, first_name)
    if missing:
        problems.append(
            Problem(
                task_id,
                "code",
                f"метрика {name!r}: функция {first_name!r} использует имена {', '.join(repr(x) for x in missing)}, "
                f"которых не будет в её globals при оценке, — вызов упадёт с NameError",
            )
        )
    return problems


def check_metric_name(task_id: str, index: int, value: object) -> tuple[str, list[Problem]]:
    name = str(value).strip()
    if not isinstance(value, str):
        return name, [Problem(task_id, "metric-name", f"TMC-list[{index}]: metric не является строкой")]
    if not name:
        return name, [Problem(task_id, "metric-name", f"TMC-list[{index}]: пустое имя метрики")]
    if name == RESERVED_METRIC:
        return name, [
            Problem(
                task_id,
                "metric-name",
                f"TMC-list[{index}]: имя {RESERVED_METRIC!r} зарезервировано, "
                f"оценка получит CREvaluator без тестовой функции и упадёт",
            )
        ]
    return name, []


def check_ground_truth(root: str, task_id: str, name: str, value: object) -> list[Problem]:
    task_dir = os.path.abspath(os.path.join(root, DATA_DIR, task_id))
    gt_dir = os.path.join(task_dir, "gt")
    target = os.path.abspath(os.path.join(gt_dir, str(value)))
    if not target.startswith(task_dir + os.sep) and target != task_dir:
        return [
            Problem(task_id, "ground-truth", f"метрика {name!r}: путь {value!r} выходит за пределы data/{task_id}/")
        ]
    if os.path.isdir(target):
        if target == gt_dir:
            return []
        return [
            Problem(task_id, "ground-truth", f"метрика {name!r}: путь {value!r} указывает на каталог, а не на файл")
        ]
    if not os.path.isfile(target):
        return [Problem(task_id, "ground-truth", f"метрика {name!r}: нет файла gt/{value}")]
    return []


def check_evaluator(task_id: str, name: str, item: dict) -> list[Problem]:
    evaluator = TM_2_EVALUATOR.get(item["metric"])
    if evaluator is not None and evaluator is not CREvaluator:
        return [
            Problem(
                task_id,
                "evaluator",
                f"метрика {name!r} попадает на {evaluator.__name__}, который оценка вызывает без "
                f"обязательного аргумента output, — результат будет нулевым при любом ключе rule",
            )
        ]
    return []


def check_tmc_item(root: str, task_id: str, index: int, item: object) -> list[Problem]:
    if not isinstance(item, dict):
        return [Problem(task_id, "metric-schema", f"TMC-list[{index}] не является словарём")]
    missing = sorted(TMC_KEYS - set(item))
    if missing:
        return [Problem(task_id, "metric-schema", f"TMC-list[{index}]: нет ключей {', '.join(missing)}")]

    name, problems = check_metric_name(task_id, index, item["metric"])
    problems += check_code(task_id, name, item["code"])
    problems += check_evaluator(task_id, name, item)

    ground_truth = item.get("ground_truth")
    if ground_truth is not None:
        problems += check_ground_truth(root, task_id, name, ground_truth)
    return problems


def check_metric(root: str, task_id: str) -> list[Problem]:
    path = os.path.join(root, METRIC_DIR, task_id, "metric.yaml")
    if not os.path.isfile(path):
        return [Problem(task_id, "metric-file", "нет файла metric.yaml")]
    try:
        config = load_yaml(path)
    except Exception as error:
        return [Problem(task_id, "metric-file", f"metric.yaml не парсится: {error}")]
    if not isinstance(config, dict):
        return [Problem(task_id, "metric-schema", "metric.yaml содержит не объект")]
    tmc_list = config.get("TMC-list")
    if not isinstance(tmc_list, list) or not tmc_list:
        return [Problem(task_id, "metric-schema", "в metric.yaml нет непустого списка TMC-list")]

    problems: list[Problem] = []
    seen: set[str] = set()
    for index, item in enumerate(tmc_list):
        problems += check_tmc_item(root, task_id, index, item)
        if isinstance(item, dict) and "metric" in item:
            key = str(item["metric"])
            if key in seen:
                problems.append(
                    Problem(
                        task_id, "metric-duplicate", f"метрика {key.strip()!r} повторяется и будет потеряна при оценке"
                    )
                )
            seen.add(key)
    return problems


def validate_task(root: str, task_id: str) -> list[Problem]:
    return check_prompt(root, task_id) + check_gt(root, task_id) + check_metric(root, task_id)


def validate(root: str, task_id: str) -> tuple[int, list[Problem]]:
    problems = check_root(root)
    all_ids, layout_problems = check_layout(root)
    if task_id == "all":
        target_ids = all_ids
        problems += layout_problems
    elif task_id in all_ids:
        target_ids = [task_id]
        problems += [problem for problem in layout_problems if problem.task_id == task_id]
    else:
        problems.append(Problem(task_id, "layout", "задача не найдена ни в data/, ни в metric/"))
        return 0, problems

    for current in target_ids:
        problems += validate_task(root, current)
    if not target_ids:
        problems.append(Problem(ROOT_ID, "root", f"не проверено ни одной задачи в {os.path.abspath(root)}"))
    return len(target_ids), problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Валидатор целостности бенчмарка DataSciBench")
    parser.add_argument("--task-id", type=str, default="all", help="Идентификатор задачи, по умолчанию all")
    parser.add_argument("--root", type=str, default=".", help="Корень бенчмарка с папками data/ и metric/")
    parser.add_argument("--json", action="store_true", help="Машиночитаемый вывод")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checked, problems = validate(args.root, args.task_id)
    if args.json:
        print(json.dumps({"checked": checked, "problems": [asdict(p) for p in problems]}, ensure_ascii=False, indent=2))
    else:
        print(f"Проверено задач: {checked}, найдено проблем: {len(problems)}")
        for problem in problems:
            print(f"  {problem.task_id} [{problem.check}] {problem.message}")
    return 1 if problems or checked == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
